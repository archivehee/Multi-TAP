

import os
import json
import time
import pickle
import argparse
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from openai import OpenAI
from config import OPENAI_API_KEY



PROFILE_ORDER = [
    "price_centric",
    "quality_score_centric",
    "popularity_centric",
    "category_preference",
    "category_diversity",
]


client = OpenAI(api_key=OPENAI_API_KEY)



def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_pkl(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def zero_vec(dim: int) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)

_last_call_ts = 0.0
def get_embeddings_batch(
    client: OpenAI,
    texts: List[str],
    model: str,
    dim: int,
    rpm: int,
    retry: int = 2,
    sleep_between_retry: float = 0.7,
) -> List[np.ndarray]:
    global _last_call_ts
    if not texts:
        return []

    proc = [(t or "").strip() for t in texts]
    empty_mask = [len(t) == 0 for t in proc]
    inputs = [t if t else " " for t in proc]

    min_interval = 60.0 / max(1, rpm)
    now = time.time()
    elapsed = now - _last_call_ts
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    for attempt in range(retry + 1):
        try:
            resp = client.embeddings.create(
                model=model,
                input=inputs,
                dimensions=dim,
                encoding_format="float",
            )
            _last_call_ts = time.time()

            outs: List[np.ndarray] = []
            for i, d in enumerate(resp.data):
                if empty_mask[i]:
                    outs.append(zero_vec(dim))
                else:
                    v = np.asarray(d.embedding, dtype=np.float32)
                    if v.shape[0] != dim:
                        if v.shape[0] > dim:
                            v = v[:dim].astype(np.float32, copy=False)
                        else:
                            buf = np.zeros((dim,), dtype=np.float32)
                            buf[: v.shape[0]] = v
                            v = buf
                    outs.append(v)
            return outs

        except Exception as e:
            tqdm.write(f"[warn] batch embedding error (attempt {attempt+1}): {e}")
            if attempt < retry:
                time.sleep(sleep_between_retry)
            else:
                return [zero_vec(dim) for _ in texts]


def iter_user_personas(records: List[dict]) -> List[Dict[str, Any]]:

    out = []
    for rec in records:
        uid = rec.get("User ID")
        if not uid:
            continue
        profiles = (rec.get("Profiles") or {}) if isinstance(rec.get("Profiles"), dict) else {}
        vec_texts = []
        for key in PROFILE_ORDER:
            persona = ""
            if key in profiles and isinstance(profiles[key], dict):
                persona = (profiles[key].get("persona") or "").strip()
            vec_texts.append(persona)
        out.append({"user_id": str(uid), "personas": vec_texts})
    return out


import glob
import os

def process_one_file(in_path, out_path, args):
    client = OpenAI(api_key=OPENAI_API_KEY)
    data = load_json(in_path)
    users = iter_user_personas(data)
    tqdm.write(f"[load] {os.path.basename(in_path)} users: {len(users)}")

    result = {
        "meta": {
            "version": "v1",
            "model": args.model,
            "dim": args.dim,
            "dtype": "float32",
            "profile_order": PROFILE_ORDER,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_json": os.path.abspath(in_path),
            "normalize_l2": bool(args.l2norm),
        },
        "embeddings": {}
    }

    B = args.batch
    all_items = [(u["user_id"], u["personas"]) for u in users]

    for i in tqdm(range(0, len(all_items), B), desc=f"Embedding:{os.path.basename(in_path)}", unit="batch"):
        chunk = all_items[i:i+B]
        flat_texts = []
        for uid, persona_list in chunk:
            flat_texts.extend(persona_list)

        embs = get_embeddings_batch(
            client=client,
            texts=flat_texts,
            model=args.model,
            dim=args.dim,
            rpm=args.rpm,
            retry=args.retry,
            sleep_between_retry=args.retry_sleep,
        )

        k = 0
        for uid, persona_list in chunk:
            vec5 = np.stack(embs[k:k+5], axis=0).astype(np.float32, copy=False)
            k += 5
            if args.l2norm:
                eps = 1e-12
                nrm = np.linalg.norm(vec5, axis=1, keepdims=True)
                vec5 = vec5 / np.clip(nrm, eps, None)

            result["embeddings"][uid] = {
                "vecs": vec5,
                "has_text": [1 if len(t.strip()) else 0 for t in persona_list],
            }

    save_pkl(out_path, result)
    tqdm.write(f"[done] saved {len(result['embeddings'])} users → {out_path}")


def main(args):
    in_path = args.in_path
    if os.path.isdir(in_path):
        json_paths = sorted(glob.glob(os.path.join(in_path, "*.json")))
        assert json_paths, f"No JSON files in: {in_path}"
        out_dir = args.out_path if os.path.isdir(args.out_path) else os.path.dirname(args.out_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        for jp in json_paths:
            base = os.path.splitext(os.path.basename(jp))[0]  # e.g., personas_Electronics
            domain = base.split("personas_")[-1]
            op = os.path.join(out_dir, f"{domain}_user_persona_emb.pkl")
            process_one_file(jp, op, args)
    else:
        out_path = args.out_path
        if os.path.isdir(out_path):
            base = os.path.splitext(os.path.basename(in_path))[0]
            domain = base.split("personas_")[-1]
            out_path = os.path.join(out_path, f"{domain}_user_persona_emb.pkl")
        process_one_file(in_path, out_path, args)

# -------------------------
# CLI
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default= "./profiles/persona_sentences/home_toys", help="input personas_*.json")
    p.add_argument("--out", dest="out_path", default= "./data/amazon/home_toys/txt_usr_emb", help="output PKL path")
    p.add_argument("--model", default="text-embedding-3-large")
    p.add_argument("--dim", type=int, default=3072)
    p.add_argument("--batch", type=int, default=128, help="users per batch (API call uses batch*5 texts)")
    p.add_argument("--rpm", type=int, default=5000, help="requests per minute (batch calls)")
    p.add_argument("--retry", type=int, default=2)
    p.add_argument("--retry-sleep", dest="retry_sleep", type=float, default=0.7)
    p.add_argument("--l2norm", action="store_true", help="apply L2 normalization per vector")
    p.add_argument("--autosave", action="store_true")
    p.add_argument("--autosave-every", type=int, default=10, help="autosave every N batches")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
    
