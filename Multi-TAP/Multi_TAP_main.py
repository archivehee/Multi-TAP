import argparse
import os
import json
import sys
import time
from pathlib import Path
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.Multi_TAP.utils.helper import print_config
from model.Multi_TAP.utils.data import load_domain, BPRDataset
from model.Multi_TAP.trainer.trainer import AltTAPTrainer
from tqdm.auto import tqdm

COSINE_MIN_LR = 1e-6

def build_parser():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--dataset_root", type=str, default="./data/amazon")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--persona_pkl", type=str, required=True)
    parser.add_argument("--source_persona_pkl", type=str, default=None,
                        help="Persona embedding table for source domain (InfoNCE).")
    parser.add_argument("--target_persona_pkl", type=str, default=None,
                        help="Override persona_pkl for target domain (InfoNCE).")
    parser.add_argument("--lgn_user_pkl", type=str, required=True)
    parser.add_argument("--source_lgn_user_pkl", type=str, default=None,
                        help="LightGCN user embedding table for source domain (InfoNCE).")
    parser.add_argument("--target_lgn_user_pkl", type=str, default=None,
                        help="LightGCN user embedding table for target domain (InfoNCE).")
    parser.add_argument("--map_dir", type=str, required=True)
    parser.add_argument("--target_map_dir", type=str, default=None,
                        help="Path to target domain user2id/item2id maps (defaults to --map_dir).")
    parser.add_argument("--source_map_dir", type=str, default=None,
                        help="Path to source domain user2id/item2id maps.")
    parser.add_argument("--lgn_item_pkl", type=str, required=True)
    parser.add_argument("--txt_item_pkl", type=str, required=True)
    parser.add_argument("--source_lgn_item_pkl", type=str, default=None,
                        help="LightGCN item embedding table for source domain.")
    parser.add_argument("--source_txt_item_pkl", type=str, default=None,
                        help="Text item embedding table for source domain.")
    parser.add_argument("--source_domain", type=str, default=None,
                        help="Source domain name for the auxiliary BPR pretraining stage.")
    # Model/training
    parser.add_argument("--dropout", type=float, default=0.5) 
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--weight_decay", type=float, default=1e-5) 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024) 
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="true")
    parser.add_argument("--neg_num", type=int, default=3, help="Number of negative samples per positive in BPR training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--info_nce_tau", type=float, default=0.4, help="InfoNCE temperature.")
    parser.add_argument("--info_nce_weight", type=float, default=1.3, help="Loss weight for InfoNCE term.")
    parser.add_argument("--K_list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--es_min_delta", type=float, default=1e-4) # 0.0001
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_check_every", type=int, default=2)    # validate every N epochs
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to store result JSON files.")
    return parser

def snapshot_model(model):
    best_params  = [p.detach().cpu().clone() for p in model.parameters()]
    best_buffers = [b.detach().cpu().clone() for b in model.buffers()]
    return best_params, best_buffers

def restore_model(model, best_params, best_buffers, device):
    for p, s in zip(model.parameters(), best_params):
        p.data.copy_(s.to(device))
    for b, s in zip(model.buffers(), best_buffers):
        b.data.copy_(s.to(device))


def pretrain_with_early_stopping(
    trainer,
    train_dl,
    domain_key,
    stage_name,
    args,
    eval_user_pos=None,
    eval_user_pos_train=None,
    item_max=None,
):
    if train_dl is None:
        print(f"[{stage_name}] No data available; skipping pretraining.")
        return
    if not (eval_user_pos and item_max):
        print(f"[{stage_name}] No validation info; cannot run early-stopping pretraining.")
        return

    best_ndcg = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_params = None
    best_buffers = None
    disable_pbar = not sys.stderr.isatty()

    print(f"[{stage_name}] BPR-only training with early stopping (domain={domain_key})")
    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        pbar = tqdm(
            train_dl,
            desc=f"[{stage_name}] Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3,
            disable=disable_pbar,
        )
        running = 0.0
        for batch in pbar:
            loss = trainer.update(batch, domain=domain_key, use_info_nce=False)
            running += loss
            avg_loss = running / max(1, pbar.n)
            pbar.set_postfix({"loss": f"{loss:.4f}", "avg": f"{avg_loss:.4f}"})
        print(f"[{stage_name}] epoch {epoch} avg_loss={(running/max(1,len(train_dl))):.4f}")

        if epoch % args.es_check_every == 0:
            metrics, ndcg20 = trainer.evaluate_full_ranking(
                eval_user_pos=eval_user_pos,
                user_pos_train=eval_user_pos_train or {},
                item_max=item_max,
                batch_size=args.batch_size,
                K_list=args.K_list,
                domain=domain_key,
            )
            print(f"[{stage_name}] valid NDCG@20={ndcg20:.6f}")
            if ndcg20 > best_ndcg + args.es_min_delta:
                best_ndcg = ndcg20
                best_epoch = epoch
                epochs_no_improve = 0
                best_params, best_buffers = snapshot_model(trainer.model)
                print(f"  ↳ improved (best NDCG@20={best_ndcg:.6f})")
            else:
                epochs_no_improve += 1
                print(f"  ↳ no improvement ({epochs_no_improve}/{args.es_patience})")
                if epochs_no_improve >= args.es_patience:
                    print(f"[{stage_name}] early stopping at epoch {epoch} (best={best_epoch})")
                    break
    if best_params is not None:
        restore_model(trainer.model, best_params, best_buffers, device=trainer.device)
    trainer.optimizer.zero_grad(set_to_none=True)


def main():
    args = build_parser().parse_args()

    def _parse_bool(value: str) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n", "none", "null"}:
            return False
        return True

    args.cuda = _parse_bool(args.cuda)

    run_start = time.time()

    # Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    (
        train_pairs,
        valid_pairs,
        test_pairs,
        user_max,
        item_max,
        user_pos_train,
        items_all,
        user_pos_valid,
        user_pos_test,
    ) = load_domain(args.dataset_root, args.domain, seed=args.seed)

    source_train_dl = None
    src_user_pos_train = None
    src_items_all = None
    src_user_pos_valid = None
    src_item_max = None
    if args.source_domain:
        if not (args.source_lgn_item_pkl and args.source_txt_item_pkl):
            raise ValueError("Source domain training requires --source_lgn_item_pkl and --source_txt_item_pkl.")
        (
            src_train_pairs,
            src_valid_pairs,
            src_test_pairs,
            src_user_max,
            src_item_max,
            src_user_pos_train,
            src_items_all,
            src_user_pos_valid,
            src_user_pos_test,
        ) = load_domain(args.dataset_root, args.source_domain, seed=args.seed)
        source_train_dl = DataLoader(
            BPRDataset(src_train_pairs, src_user_pos_train, src_items_all, neg_num=args.neg_num),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

    target_map_dir = args.target_map_dir or args.map_dir
    source_map_dir = args.source_map_dir

    cfg = {
        "persona_pkl": args.persona_pkl,
        "source_persona_pkl": args.source_persona_pkl,
        "target_persona_pkl": args.target_persona_pkl,
        "lgn_user_pkl": args.lgn_user_pkl,
        "source_lgn_user_pkl": args.source_lgn_user_pkl,
        "target_lgn_user_pkl": args.target_lgn_user_pkl,
        "map_dir": args.map_dir,
        "target_map_dir": target_map_dir,
        "source_map_dir": source_map_dir,
        "lgn_item_pkl": args.lgn_item_pkl,
        "txt_item_pkl": args.txt_item_pkl,
        "source_lgn_item_pkl": args.source_lgn_item_pkl,
        "source_txt_item_pkl": args.source_txt_item_pkl,
        "aggregator": "self_attention",
        "dropout": args.dropout,
        "optim": args.optim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"),
        "train_persona_emb": True,
        "train_lgn_emb": True,
        "train_item_emb": True,
        "use_info_nce": True,
        "info_nce_mode": "doppelganger",
        "info_nce_tau": args.info_nce_tau,
        "info_nce_weight": args.info_nce_weight,
        "info_nce_detach_persona": True,
    }
    print_config({**cfg,
                  "domain": args.domain,
                  "user_max": user_max,
                  "item_max": item_max,
                  "neg_num": args.neg_num,
                  "lr_scheduler": args.lr_scheduler,
                  "source_domain": args.source_domain})

    # Trainer
    trainer = AltTAPTrainer(cfg)

    # DataLoaders
    train_dl = DataLoader(
        BPRDataset(train_pairs, user_pos_train, items_all, neg_num=args.neg_num),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.cuda
    )

    if source_train_dl is not None:
        pretrain_with_early_stopping(
            trainer,
            source_train_dl,
            "source",
            "Pretrain-Source",
            args,
            eval_user_pos=src_user_pos_valid,
            eval_user_pos_train=src_user_pos_train,
            item_max=src_item_max,
        )
    elif args.source_domain:
        print("[warn] source_domain is set, but no training data is available; skipping pretraining.")

    if args.source_domain:
        trainer.model.freeze_source_persona_transform()
        print("[Info] Frozen source-domain self-attention/FFN/projection parameters.")

    if args.lr_scheduler == "cosine":
        total_steps = max(1, len(train_dl) * args.epochs)
        trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=total_steps, eta_min=COSINE_MIN_LR)


    best_ndcg20 = -1.0
    best_epoch = -1
    epochs_no_improve = 0


    best_params = None
    best_buffers = None
    
    disable_pbar = not sys.stderr.isatty()
    for epoch in range(1, args.epochs + 1):
        trainer.model.train()
        pbar = tqdm(
            train_dl,
            desc=f"Epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3,
            disable=disable_pbar,
        )
        running = 0.0
        for batch in pbar:
            loss = trainer.update(batch)
            running += loss
            pbar.set_postfix({"loss": f"{loss:.4f}", "avg": f"{(running/max(1,pbar.n)):.4f}"})

        # Log at the end of each epoch
        print(f"[Epoch {epoch}] avg_loss={(running/max(1,len(train_dl))):.4f}")

        # Validate every N epochs
        if epoch % args.es_check_every == 0:
            metrics, ndcg20 = trainer.evaluate_full_ranking(
                eval_user_pos=user_pos_valid,
                user_pos_train=user_pos_train,
                item_max=item_max,
                batch_size=args.batch_size,
                K_list=args.K_list,
                domain="target"
            )

            print(f"[Valid @ epoch {epoch}] NDCG@20={ndcg20:.6f}")
            

            # Early stopping check
            if ndcg20 > best_ndcg20 + args.es_min_delta:
                best_ndcg20 = ndcg20
                best_epoch = epoch
                epochs_no_improve = 0
                best_params, best_buffers = snapshot_model(trainer.model)
                print(f"  ↳ better results (best NDCG@20={best_ndcg20:.6f} @ epoch {best_epoch})")
            else:
                epochs_no_improve += 1
                print(f"  ↳ no improvement ({epochs_no_improve}/{args.es_patience} patience without improvement)")
                if epochs_no_improve >= args.es_patience:
                    print(f"early stopping: best_epoch={best_epoch}, best_ndcg20={best_ndcg20:.6f}")
                    break
            
    if best_params is not None:
        restore_model(trainer.model, best_params, best_buffers, device=cfg["device"])
        print(f"Testing uses best_epoch={best_epoch} (NDCG@20={best_ndcg20:.6f}).")

    test_metrics, _ = trainer.evaluate_full_ranking(
        eval_user_pos=user_pos_test,
        user_pos_train=user_pos_train,
        item_max=item_max,
        batch_size=args.batch_size,
        K_list=args.K_list,
        domain="target"
    )


    elapsed_sec = time.time() - run_start
    unique_users = set(user_pos_train.keys()) | set(user_pos_valid.keys()) | set(user_pos_test.keys())
    test_metrics["_meta"] = {
        "total_time_sec": float(elapsed_sec),
        "total_time_min": float(elapsed_sec / 60.0),
        "users_kept": int(len(unique_users)),
        "seed": int(args.seed),
    }
    print(f"[Time] total_time_sec={elapsed_sec:.2f} (users_kept={len(unique_users)})")

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = os.path.join(args.output_dir, f"Multi_TAP_res_{args.domain}.json")
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"[Test] Saved results to: {out_name}")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
