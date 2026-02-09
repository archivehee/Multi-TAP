from typing import Dict, List, Tuple, Set
import os
import random
import torch
from torch.utils.data import Dataset
 
 
def _read_pairs(path: str) -> List[Tuple[int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            u, i = line.strip().split("\t")[:2]
            pairs.append((int(u), int(i)+1))  # Items are 1-based (0 reserved for padding)
    return pairs

def _build_user_pos(pairs: List[Tuple[int, int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for u, i in pairs:
        out.setdefault(u, set()).add(i)
    return out


def load_domain(root: str, domain: str, user_frac: float = 1.0, seed: int = 0, min_users: int = 1):

    dpath = os.path.join(root, domain)
    tr = _read_pairs(os.path.join(dpath, "train.txt"))
    va = _read_pairs(os.path.join(dpath, "valid.txt")) if os.path.exists(os.path.join(dpath,"valid.txt")) else []
    te = _read_pairs(os.path.join(dpath, "test.txt"))
    if user_frac < 1.0:
        all_users = sorted({u for u, _ in tr + va + te})
        if all_users:
            k = int(len(all_users) * user_frac)
            k = max(min_users, k)
            rng = random.Random(seed)
            selected = set(rng.sample(all_users, min(k, len(all_users))))
            tr = [p for p in tr if p[0] in selected]
            va = [p for p in va if p[0] in selected]
            te = [p for p in te if p[0] in selected]
    user_max = max([u for u,_ in tr + va + te] + [0]) + 1
    item_max = max([i for _,i in tr + va + te] + [0]) + 1
    user_pos_train = _build_user_pos(tr)
    user_pos_valid = _build_user_pos(va)
    user_pos_test = _build_user_pos(te)
    items_all = set(i for _, i in tr + va + te)
    return (
        tr,
        va,
        te,
        user_max,
        item_max,
        user_pos_train,
        items_all,
        user_pos_valid,
        user_pos_test,
    )


class BPRDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int,int]], user_pos: Dict[int,set], items_all: set, neg_num: int = 1):
        self.pairs = pairs
        self.user_pos = user_pos
        self.items_all = list(items_all)
        self.neg_num = max(1, neg_num)

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos = self.pairs[idx]
        negatives = []
        tried = 0
        max_trials = 50 * self.neg_num
        while len(negatives) < self.neg_num and tried < max_trials:
            tried += 1
            neg = random.choice(self.items_all)
            if neg == pos:
                continue
            if neg in self.user_pos.get(u, set()):
                continue
            if neg in negatives:
                continue
            negatives.append(neg)
        if len(negatives) < self.neg_num:
            # Fill the remainder randomly, allowing duplicates
            while len(negatives) < self.neg_num:
                neg = random.choice(self.items_all)
                if neg != pos:
                    negatives.append(neg)

        neg_tensor = torch.tensor(negatives, dtype=torch.long)
        return (torch.tensor(u).long(),
                torch.tensor(pos).long(),
                neg_tensor)


class EvalDataset(Dataset):
    def __init__(self,
                 user_pos_eval: Dict[int, set],
                 user_pos_train: Dict[int, set],
                 items_all: set,
                 K: int = 100):
        self.users = list(user_pos_eval.keys())
        self.user_pos_eval = user_pos_eval
        self.user_pos_train = user_pos_train
        self.items_all = list(items_all)
        self.base_K = K
        self.max_pos = max((len(v) for v in user_pos_eval.values()), default=0)
        self.cand_size = max(self.base_K, self.max_pos)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_set = self.user_pos_eval[u]
        cand = [(item, True) for item in pos_set]
        pos_train = self.user_pos_train.get(u, set())
        while len(cand) < self.cand_size:
            neg = random.choice(self.items_all)
            if neg in pos_set or neg in pos_train:
                continue
            cand.append((neg, False))
        random.shuffle(cand)
        items = torch.tensor([item for item, _ in cand], dtype=torch.long)
        pos_mask = torch.tensor([flag for _, flag in cand], dtype=torch.bool)
        return torch.tensor(u).long(), items, pos_mask
