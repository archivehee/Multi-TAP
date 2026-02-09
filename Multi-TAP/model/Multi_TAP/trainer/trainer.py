import torch
from .Multi_TAP import AltTAPModel
from ..utils.torch_utils import get_optimizer, change_lr
from ..utils.metrics import finalize_metrics, accumulate_user_metrics


class AltTAPTrainer:
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.model = AltTAPModel(cfg).to(self.device)
        # Optimizer uses only parameters with requires_grad=True
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_optimizer(
            cfg["optim"],
            params,
            lr=cfg["lr"],
            l2=cfg["weight_decay"],
        )
        self.scheduler = None


    def update(self, batch, domain: str = None, use_info_nce: bool = None) -> float:
        self.model.train()
        uid, pos_iid, neg_iid = batch
        uid = uid.to(self.device)
        pos_iid = pos_iid.to(self.device)
        neg_iid = neg_iid.to(self.device)
        contrastive_uid = uid

        if neg_iid.dim() == 2:
            B, K = neg_iid.shape
            uid = uid.unsqueeze(1).expand(-1, K).reshape(-1)
            pos_iid = pos_iid.unsqueeze(1).expand(-1, K).reshape(-1)
            neg_flat = neg_iid.reshape(-1)
        else:
            neg_flat = neg_iid.view(-1)

        loss = self.model.bpr_loss(uid, pos_iid, neg_flat, domain=domain)
        info_active = self.model.use_info_nce if use_info_nce is None else (use_info_nce and self.model.use_info_nce)
        if info_active:
            contrastive_loss = self.model.info_nce_loss(contrastive_uid)
            loss = loss + self.model.info_nce_weight * contrastive_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.item())

    def update_lr(self, new_lr: float):
        change_lr(self.optimizer, new_lr)
        
        
    def evaluate_full_ranking(self, eval_user_pos, user_pos_train, item_max,
                            batch_size=1024, K_list=(5,10,15,20,25,30,35,40,45,50),
                            domain: str = None):

        self.model.eval()

        # 1) Cache all item embeddings [I-1, 256]
        all_item_emb = self.model.all_item_matrix(item_max, self.device, domain=domain)  # [I-1,256]

        agg_total = {K: {"hit":0.0,"ndcg":0.0,"prec":0.0,"rec":0.0,"f1":0.0} for K in K_list}
        user_total, auc_total = 0, 0.0

        users = list(eval_user_pos.keys())
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start+batch_size]
            uid = torch.tensor(batch_users, device=self.device, dtype=torch.long)
            u_emb = self.model.user_enc(uid, domain=domain)                    # [B,256]
            scores_all = torch.matmul(u_emb, all_item_emb.T)    # [B, I-1]

            pos_mask = torch.zeros(scores_all.size(), dtype=torch.bool, device=self.device)
            for b, user in enumerate(batch_users):
                pos_set = eval_user_pos[user]
                if not pos_set:
                    continue
                pos_indices = [item - 1 for item in pos_set if 1 <= item < item_max]
                if pos_indices:
                    pos_mask[b, torch.tensor(pos_indices, device=self.device)] = True
                seen = user_pos_train.get(user, set())
                if seen:
                    seen_indices = [item - 1 for item in seen if 1 <= item < item_max and item not in pos_set]
                    if seen_indices:
                        scores_all[b, torch.tensor(seen_indices, device=self.device)] = float("-inf")

            agg, users_count, auc_sum = accumulate_user_metrics(scores_all, pos_mask, K_list)
            for K in K_list:
                for key in agg[K]:
                    agg_total[K][key] += agg[K][key]
            user_total += users_count
            auc_total += auc_sum

        metrics = finalize_metrics(agg_total, user_total, auc_total, K_list)
        ndcg20 = metrics.get("K=20", {}).get("NDCG", 0.0)
        return metrics, float(ndcg20)
