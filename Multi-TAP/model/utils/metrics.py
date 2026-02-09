import math
import torch


def _ideal_dcg(pos_cnt: int, K: int) -> float:
    limit = min(pos_cnt, K)
    if limit <= 0:
        return 0.0
    return sum(1.0 / math.log2(i + 2.0) for i in range(limit))


@torch.no_grad()
def accumulate_user_metrics(scores: torch.Tensor,
                            pos_mask: torch.Tensor,
                            K_list=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50)):
    """
    scores:   [B, L]  candidate item scores
    pos_mask: [B, L]  True where the item is a positive for the user
    """
    device = scores.device
    B, L = scores.shape
    agg = {K: {"hit": 0.0, "ndcg": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0} for K in K_list}
    auc_total = 0.0
    user_total = 0

    for b in range(B):
        mask = pos_mask[b]
        pos_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        pos_cnt = pos_idx.numel()
        if pos_cnt == 0:
            continue

        score = scores[b]
        # Rank computation
        sorted_indices = torch.argsort(score, descending=True)
        ranks = torch.empty_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(L, device=device)
        pos_ranks = ranks[pos_idx]
        pos_ranks_float = pos_ranks.float()

        for K in K_list:
            hits = (pos_ranks < K).sum().item()
            hr = 1.0 if hits > 0 else 0.0
            rec = hits / float(pos_cnt)
            prec = hits / float(K)

            if hits > 0:
                dcg = float((1.0 / torch.log2(pos_ranks_float[pos_ranks < K] + 2.0)).sum().item())
            else:
                dcg = 0.0
            idcg = _ideal_dcg(pos_cnt, K)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)

            agg[K]["hit"] += hr
            agg[K]["rec"] += rec
            agg[K]["prec"] += prec
            agg[K]["ndcg"] += ndcg
            agg[K]["f1"] += f1

        # AUC
        neg_mask = ~mask
        neg_scores = score[neg_mask]
        if neg_scores.numel() > 0:
            pos_scores = score[pos_idx]
            cmp = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float()
            ties = (pos_scores.unsqueeze(1) == neg_scores.unsqueeze(0)).float()
            auc = (cmp.sum() + 0.5 * ties.sum()) / (pos_scores.numel() * neg_scores.numel())
            auc_total += float(auc.item())
        user_total += 1

    return agg, user_total, auc_total


def finalize_metrics(agg, user_count, auc_sum, K_list):
    out = {}
    denom = max(1, user_count)
    for K in K_list:
        out[f"K={K}"] = {
            "HR":        agg[K]["hit"]  / denom,
            "NDCG":      agg[K]["ndcg"] / denom,
            "Precision": agg[K]["prec"] / denom,
            "Recall":    agg[K]["rec"]  / denom,
            "F1":        agg[K]["f1"]   / denom,
            "AUC":       auc_sum / denom,
        }
    return out
