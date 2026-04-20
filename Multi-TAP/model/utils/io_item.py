from typing import Dict, Tuple
import os, pickle, numpy as np, torch

def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_item2id(map_dir: str) -> Dict[str, int]:
    # maps/item2id.txt : "<ASIN>\t<id>"
    mp = {}
    with open(os.path.join(map_dir, "item2id.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            asin, idx = line.strip().split("\t")[:2]
    mp[asin] = int(idx) + 1  # 1-based (0 is padding)
    return mp

def _get_vec_from_dict(d, keys, dim) -> np.ndarray:
    as_dict = dict(d.items()) if isinstance(d, dict) else {}
    for k in keys:
        if k not in as_dict or as_dict[k] is None:
            continue
        v = as_dict[k]
        if isinstance(v, str):
            # Ignore raw labels
            continue
        try:
            arr = np.asarray(v, dtype=np.float32)
        except ValueError:
            continue
        if arr.ndim == 1 and arr.shape[0] == dim and np.isfinite(arr).all():
            return arr
    return np.zeros((dim,), dtype=np.float32)

def build_item_feature_table(
    map_dir: str,
    lgn_item_pkl: str,
    txt_item_pkl: str,
    lgn_dim: int = 128,
    txt_dim: int = 512
) -> torch.Tensor:
    """
    Returns: torch.FloatTensor [num_items+1, 1664]  (row 0 is zero padding)
    txt_item_pkl expected formats:
      { "<ASIN>": {"desc": 512, "keywords": 512, "category": 512}, ... }
      Also supports key variants: ("desc","description"), ("keywords","kw"), ("category","cat")
    lgn_item_pkl formats: dict[str->128], dict[int->128], or np.array[num,128]
    """
    asin2id = load_item2id(map_dir)
    num_items = max(asin2id.values())  # already 1-based
    out = np.zeros((num_items + 1, lgn_dim + 3 * txt_dim), dtype=np.float32)

    # --- LGN ---
    lgn_obj = _load_pkl(lgn_item_pkl)
    # Handle supported formats
    if isinstance(lgn_obj, dict):
        # Key is str (asin) or int (id)
        ex_key = next(iter(lgn_obj.keys()))
        if isinstance(ex_key, str):  # asin key
            for asin, idx in asin2id.items():
                v = lgn_obj.get(asin)
                if v is not None:
                    out[idx, :lgn_dim] = np.asarray(v, dtype=np.float32)
        else:  # id key
            for asin, idx in asin2id.items():
                v = lgn_obj.get(idx - 1) or lgn_obj.get(idx)
                if v is not None:
                    out[idx, :lgn_dim] = np.asarray(v, dtype=np.float32)
    elif isinstance(lgn_obj, np.ndarray):
        # id-based
        for asin, idx in asin2id.items():
            if 0 <= idx < lgn_obj.shape[0]:
                out[idx, :lgn_dim] = lgn_obj[idx]
    else:
        raise ValueError("Unsupported LGN item pkl format")

    # --- TEXT ---
    txt_obj = _load_pkl(txt_item_pkl)
    # Expected formats:
    #   1) {asin: {desc/keywords/category: 512d}, ...}
    #   2) {asin: {"embedding": 1536d, "category": "raw"}, ...}
    #   3) {"embeddings": {...}} (nested)
    if isinstance(txt_obj, dict) and "embeddings" in txt_obj:
        txt_obj = txt_obj["embeddings"]

    for asin, idx in asin2id.items():
        entry = txt_obj.get(asin, {}) if isinstance(txt_obj, dict) else {}
        if isinstance(entry, dict) and "embedding" in entry and entry.get("embedding") is not None:
            emb = np.asarray(entry["embedding"], dtype=np.float32)
            if emb.ndim == 1 and emb.shape[0] == (lgn_dim + 3 * txt_dim):
                out[idx, :] = emb
                continue
            if emb.ndim == 1 and emb.shape[0] == 3 * txt_dim:
                out[idx, lgn_dim:] = emb
                continue
        desc = _get_vec_from_dict(entry, ("desc","description"), txt_dim)
        kw   = _get_vec_from_dict(entry, ("keywords","kw"), txt_dim)
        cat  = _get_vec_from_dict(entry, ("category","cat","category_vec"), txt_dim)
        out[idx, lgn_dim : lgn_dim+txt_dim]               = desc
        out[idx, lgn_dim+txt_dim : lgn_dim+2*txt_dim]     = kw
        out[idx, lgn_dim+2*txt_dim : lgn_dim+3*txt_dim]   = cat

    return torch.from_numpy(out)
