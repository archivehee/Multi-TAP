
import os
import sys
import json
import pickle
import time
from typing import Dict, Tuple, List, Any
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import OPENAI_API_KEY


from tqdm import tqdm
from openai import OpenAI

target_pair = "home_toys"
DOMAINS = ["Home_and_Kitchen", "Toys_and_Games"]

DESC_DIR = f"./profiles/domain_desc/{target_pair}"
ITEM_META_DIR = f"./data/amazon/{target_pair}/filtered_data/f_item_meta"
MAP_DIR_BASE = f"./data/amazon/{target_pair}"

OUT_DIR = f"./data/amazon/{target_pair}/txt_itm_emb"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "text-embedding-3-large"

DESC_DIM = 512
KW_DIM   = 512
CAT_DIM  = 512
FINAL_DIM = DESC_DIM + KW_DIM + CAT_DIM  # 1536

BATCH_SIZE = 128
RPM = 5000
AUTOSAVE_EVERY = 200
RETRY = 2
SLEEP_BETWEEN_RETRY_SEC = 0.7

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_category(it: dict) -> str:
    """Use the second category if present; otherwise return "Unknown"."""
    cats = it.get("categories")
    if isinstance(cats, list) and cats:
        strings = [s.strip() for s in cats if isinstance(s, str) and s.strip()]
        if len(strings) > 1:
            return strings[1]
    return "Unknown"

def load_json_safe(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def zero_vec(dim: int) -> List[float]:
    return [0.0] * dim

def get_embedding(text: str, dim: int, label: str = "") -> List[float]:
    txt = (text or "").strip()
    if not txt:
        return zero_vec(dim)

    for attempt in range(RETRY):
        try:
            resp = client.embeddings.create(
                model=MODEL,
                input=txt[:20000],
                dimensions=dim,
                encoding_format="float",
            )
            emb = resp.data[0].embedding
            if len(emb) != dim:
                emb = emb[:dim] if len(emb) > dim else emb + [0.0]*(dim - len(emb))
            return emb
        except Exception as e:
            tqdm.write(f"[warn] embedding error ({label}) attempt {attempt+1}: {e}")
            sleep(SLEEP_BETWEEN_RETRY_SEC)
    return zero_vec(dim)

_last_call_ts = 0.0
def get_embeddings_batch(texts: List[str], dim: int, label: str = "") -> List[List[float]]:
    """
    texts: list of length N
    returns: N x dim list (empty/fail -> zero-vector)
    """
    global _last_call_ts
    min_interval = 60.0 / max(RPM, 1)

    preprocessed = [(t or "").strip() for t in texts]
    mask_empty = [len(t) == 0 for t in preprocessed]
    inputs = [t[:20000] if t else " " for t in preprocessed]

    now = time.time()
    elapsed = now - _last_call_ts
    if elapsed < min_interval:
        sleep(min_interval - elapsed)

    for attempt in range(RETRY):
        try:
            resp = client.embeddings.create(
                model=MODEL,
                input=inputs,
                dimensions=dim,
                encoding_format="float",
            )
            _last_call_ts = time.time()
            outs = []
            for i, d in enumerate(resp.data):
                emb = d.embedding
                if mask_empty[i]:
                    outs.append(zero_vec(dim))
                    continue
                if len(emb) != dim:
                    emb = emb[:dim] if len(emb) > dim else emb + [0.0]*(dim - len(emb))
                outs.append(emb)
            return outs
        except Exception as e:
            tqdm.write(f"[warn] batch embedding error ({label}) attempt {attempt+1}: {e}")
            sleep(SLEEP_BETWEEN_RETRY_SEC)

    return [zero_vec(dim) for _ in texts]

def concat3(a: List[float], b: List[float], c: List[float]) -> List[float]:
    return list(a) + list(b) + list(c)

def load_domain_profile(domain: str) -> Tuple[str, List[str]]:
    path = os.path.join(DESC_DIR, f"{domain}_desc.json")
    data = load_json_safe(path)
    prof = data.get("Domain Profile", {})
    desc = prof.get("Domain Description", "") or ""
    keywords = prof.get("Domain keywords", []) or []
    return desc, [str(k) for k in keywords if isinstance(k, (str, int, float))]

def embed_domain_once(domain: str) -> Tuple[List[float], List[float]]:
    desc, keywords = load_domain_profile(domain)
    kw_text = ", ".join(keywords) if keywords else ""
    e_desc = get_embedding(desc, DESC_DIM, label=f"{domain}/desc")
    e_kw   = get_embedding(kw_text, KW_DIM, label=f"{domain}/keywords")
    return e_desc, e_kw

def load_items(domain: str) -> List[dict]:
    in_path = os.path.join(ITEM_META_DIR, f"f_meta_{domain}.json")
    items = load_json_safe(in_path)
    if not isinstance(items, list):
        raise ValueError(f"Invalid items format: {in_path}")
    return items

def load_item2id_asins(map_dir: str) -> List[str]:
    path = os.path.join(map_dir, "item2id.txt")
    if not os.path.exists(path):
        return []
    asins = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            asin = s.split("\t")[0]
            if asin:
                asins.append(asin)
    return asins

def out_path_for(domain: str) -> str:
    return os.path.join(OUT_DIR, f"{domain}_item_text_emb.pkl")

def save_results(path: str, results: Dict[str, dict]) -> None:
    with open(path, "wb") as f:
        pickle.dump(results, f)

def load_results_if_any(path: str) -> Dict[str, dict]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                return {}
    return {}

def process_domain(domain: str):
    tqdm.write(f"\n[Domain] {domain} → embedding description/keywords (512+512)")
    e_desc_512, e_kw_512 = embed_domain_once(domain)

    items = load_items(domain)
    tqdm.write(f"[Domain] {domain} items: {len(items)}")

    map_dir = os.path.join(MAP_DIR_BASE, domain, "maps")
    keep_asins = set(load_item2id_asins(map_dir))
    if keep_asins:
        tqdm.write(f"[Align] {domain}: keep {len(keep_asins)} asins from {map_dir}/item2id.txt")
    else:
        tqdm.write(f"[Align] {domain}: no item2id.txt found; skip align")

    out_path = out_path_for(domain)
    results: Dict[str, dict] = load_results_if_any(out_path)
    tqdm.write(f"[Resume] loaded {len(results)} existing items from {out_path}")
    if keep_asins:
        before = len(results)
        results = {k: v for k, v in results.items() if k in keep_asins}
        if len(results) != before:
            tqdm.write(f"[Align] {domain}: filtered cached {before} -> {len(results)} items")

    pending_ids: List[str] = []
    pending_cats: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = it.get("parent_asin") or it.get("asin")
        if not item_id or not isinstance(item_id, str):
            continue
        if keep_asins and item_id not in keep_asins:
            continue
        if item_id in results:
            continue
        category = extract_category(it)
        pending_ids.append(item_id)
        pending_cats.append(category)

    total_pending = len(pending_ids)
    tqdm.write(f"[Plan] {domain}: to-embed {total_pending} categories (batch={BATCH_SIZE})")

    progress_bar = tqdm(total=total_pending, desc=f"{domain} items", unit="item", leave=False, dynamic_ncols=True)
    total_done = 0
    for start in range(0, total_pending, BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pending_ids))
        batch_ids = pending_ids[start:end]
        batch_cats = pending_cats[start:end]

        batch_embs = get_embeddings_batch(batch_cats, CAT_DIM, label=f"{domain}/cat_batch")
        
        if start == 0 and batch_embs:
                print(f"[Debug] Sample embedding for '{batch_cats[0]}': {batch_embs[0][:5]} ... (len={len(batch_embs[0])})")
                
        for iid, cat, e_cat_512 in zip(batch_ids, batch_cats, batch_embs):
            final_vec = concat3(e_desc_512, e_kw_512, e_cat_512)  # 1536
            if len(final_vec) != FINAL_DIM:
                final_vec = (final_vec[:FINAL_DIM]
                             if len(final_vec) > FINAL_DIM
                             else final_vec + [0.0] * (FINAL_DIM - len(final_vec)))
            results[iid] = {
                "embedding": final_vec,
                "category": cat,
            }
            total_done += 1
        progress_bar.update(len(batch_ids))

        if total_done and (total_done % AUTOSAVE_EVERY == 0):
            save_results(out_path, results)
            tqdm.write(f"[*] Auto-saved {len(results)} items → {out_path}")

    progress_bar.close()

    save_results(out_path, results)
    tqdm.write(f"[Done] {domain}: saved {len(results)} items → {out_path}")

if __name__ == "__main__":
    for dom in DOMAINS:
        process_domain(dom)
