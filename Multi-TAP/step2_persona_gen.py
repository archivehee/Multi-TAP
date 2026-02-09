import os
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import threading
import concurrent.futures

from config import OPENAI_API_KEY
from openai import OpenAI


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read().strip()
        return json.loads(txt) if txt else {}
    except:
        return {}

def extract_category(it):
    cats = it.get("categories")
    if isinstance(cats, list) and cats:
        strings = [s.strip() for s in cats if isinstance(s, str) and s.strip()]
        if len(strings) > 1:
            return strings[1]
    return "Unknown"

def _assign_price_tags_by_category(meta_map: dict):
    from collections import defaultdict
    import numpy as np

    cat_prices = defaultdict(list)
    for asin, it in meta_map.items():
        cat = it.get("category")
        price = it.get("price")
        if cat and isinstance(price, (int, float)):
            cat_prices[cat].append(float(price))

    cat_bounds = {}
    for cat, arr in cat_prices.items():
        if not arr:
            continue
        arr_np = np.array(arr, dtype=float)
        if np.all(arr_np == arr_np[0]):
            cat_bounds[cat] = None
            continue
        if len(arr_np) >= 5:
            qs = np.quantile(arr_np, [0.2, 0.4, 0.6, 0.8])
            cat_bounds[cat] = qs
        else:
            med = float(np.median(arr_np))
            eps = 1e-9
            cat_bounds[cat] = [med - eps, med, med + eps, med + 2*eps]

    for asin, it in meta_map.items():
        if isinstance(it.get("price_tag"), (int, float)):
            continue
        cat = it.get("category")
        price = it.get("price")
        if not (cat and isinstance(price, (int, float))):
            it["price_tag"] = None
            continue
        bounds = cat_bounds.get(cat)
        if bounds is None:
            it["price_tag"] = 3
            continue
        b1, b2, b3, b4 = bounds
        p = float(price)
        tag = 1
        if p > b1: tag += 1
        if p > b2: tag += 1
        if p > b3: tag += 1
        if p > b4: tag += 1
        it["price_tag"] = int(max(1, min(5, tag)))

def load_item_meta(meta_dir):
    out = {}
    for fn in os.listdir(meta_dir):
        if not fn.startswith("f_meta_") or not fn.endswith(".json"):
            continue
        domain = fn[len("f_meta_"):-5]
        items = load_json(os.path.join(meta_dir, fn)) or []
        meta_map = {}
        for it in items:
            asin = (it.get("parent_asin") or it.get("asin") or "").upper()
            if not asin:
                continue
            it["category"] = extract_category(it)
            meta_map[asin] = it
        _assign_price_tags_by_category(meta_map)
        out[domain] = meta_map
    return out

def load_selected_users(path):
    users = set()
    if not path or not os.path.exists(path):
        return users
    data = load_json(path) or []
    # Support both list and dict formats
    if isinstance(data, list):
        for obj in data:
            uid = obj.get("user_id")
            if uid:
                users.add(uid)
    elif isinstance(data, dict):
        if "user_id" in data:
            users.add(data["user_id"])
        else:
            # {uid: {...}} format
            users.update(list(data.keys()))
    return users

def time_cap(v):
    if v is None:
        return 0.0
    try:
        x = float(v)
        return x / 1000.0 if x >= 1e12 else x
    except:
        return 0.0

def load_dom_desc(desc_dir):
    out = {}
    for fn in os.listdir(desc_dir):
        if not fn.endswith("_desc.json"):
            continue
        domain = fn[:-10]
        js = load_json(os.path.join(desc_dir, fn)) or {}
        desc = (js.get("Domain Profile", {}) or {}).get("Domain Description", "")
        out[domain] = desc
    return out

def load_user_info(user_info_dir):
    out = {}
    for fn in os.listdir(user_info_dir):
        if not fn.endswith("_usr_info.json"):
            continue
        domain = fn[:-14]
        data = load_json(os.path.join(user_info_dir, fn)) or []
        mapping = {}
        if isinstance(data, list):
            for obj in data:
                uid = obj.get("user_id")
                if uid:
                    mapping[uid] = obj
        elif isinstance(data, dict):
            if "user_id" in data:
                mapping[data["user_id"]] = data
            else:
                mapping = data
        out[domain] = mapping
    return out

def _load_domain_maps_and_train(tvt_root, domain):
    maps_dir = os.path.join(tvt_root, domain, "maps")
    user2id_path = os.path.join(maps_dir, "user2id.txt")
    item2id_path = os.path.join(maps_dir, "item2id.txt")
    train_path   = os.path.join(tvt_root, domain, "train.txt")

    def _load_kv(path):
        m = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: 
                    continue
                # Assume "raw_id idx" format
                raw, idx = s.split()
                m[raw] = int(idx)
        return m

    user2idx = _load_kv(user2id_path)
    item2idx = _load_kv(item2id_path)

    # Reverse mapping (optional)
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    # Load train pairs: "u_idx i_idx" format
    train_pairs = set()
    users_in_train_idx = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            u_str, i_str = s.split()
            u_idx, i_idx = int(u_str), int(i_str)
            train_pairs.add((u_idx, i_idx))
            users_in_train_idx.add(u_idx)

    # Raw user id set (for sampling/filtering)
    users_in_train_raw = {idx2user[u] for u in users_in_train_idx if u in idx2user}

    return user2idx, item2idx, idx2user, idx2item, train_pairs, users_in_train_raw

def scan_usr_rev_train_only(review_dir, domain, allowed_raw_users, max_hist_per_user, 
                            user2idx, item2idx, train_pairs):

    out = {domain: defaultdict(list)}
    path = os.path.join(review_dir, f"{domain}.json")
    data = load_json(path) or []
    if not isinstance(data, list):
        return out

    for r in data:
        uid_raw = r.get("user_id")
        asin_raw = (r.get("parent_asin") or r.get("asin") or "").upper()
        if not uid_raw or not asin_raw:
            continue

        # If allowed users are provided, keep only those
        if allowed_raw_users and uid_raw not in allowed_raw_users:
            continue

        # raw -> index mapping
        u_idx = user2idx.get(uid_raw)
        i_idx = item2idx.get(asin_raw)
        if u_idx is None or i_idx is None:
            continue

        # Keep only interactions present in train
        if (u_idx, i_idx) not in train_pairs:
            continue

        ts = time_cap(r.get("timestamp"))
        out[domain][uid_raw].append({
            "asin": asin_raw,
            "user_rating": r.get("rating"),
            "timestamp": ts
        })

    # Keep most recent max_hist_per_user per user
    for uid, logs in out[domain].items():
        logs.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
        if max_hist_per_user is not None and len(logs) > max_hist_per_user:
            out[domain][uid] = logs[:max_hist_per_user]

    return out


def build_inputs(domain, meta, reviews, domain_descs, user_info):
    inputs = []
    m_domain = meta.get(domain, {})
    desc = domain_descs.get(domain, domain)
    uinfo_map = user_info.get(domain, {}) or {}
    for uid, logs in reviews.get(domain, {}).items():
        hist = []
        for ev in logs:
            m = m_domain.get(ev["asin"], {})
            hist.append({
                "item_id": ev["asin"],
                "title": m.get("title"),
                "category": m.get("category"),
                "average_rating": m.get("average_rating"),
                "rating_number": m.get("rating_number"),
                "user_rating": ev.get("user_rating"),
                "cat_item_price_tag": m.get("price_tag", None)
            })
        uinfo = uinfo_map.get(uid, {}) or {}
        inputs.append({
            "Domain": domain,
            "Domain description": desc,
            "User ID": uid,
            "category_price_level": uinfo.get("price_affiliated_group", {}),
            "category_rating_level": uinfo.get("rating_score_preferred_group", {}),
            "category_popularity_level": uinfo.get("rating_nums_preferred_group", {}),
            "category_familiarity_level": uinfo.get("cats_familiarity", {}),
            "overall_category_diversity": uinfo.get("cats_interaction_diversity", ""),
            "History": hist,
        })
    return inputs

class JsonArrayWriter:
    def __init__(self, out_path, meta=None, autosave_every=20, array_only=True):
        self.out_path = out_path
        self.tmp_path = out_path + ".tmp"
        self.autosave_every = autosave_every
        self.write_count = 0
        self.array_only = array_only

        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.f = open(self.tmp_path, "w", encoding="utf-8")
        self._open = True
        self._committed = False
        self.first = True

        if self.array_only:
            # Write an array (not JSONL)
            self.f.write('[\n')
        else:
            # Legacy format option (if needed)
            self.f.write('{"meta": ')
            self.f.write(json.dumps(meta or {}, ensure_ascii=False))
            self.f.write(', "records": [\n')

        self._flush_and_sync()

    def write_obj(self, obj):
        if not self._open:
            return
        if not self.first:
            self.f.write(',\n')
        self.first = False
        self.f.write(json.dumps(obj, ensure_ascii=False))
        self.write_count += 1
        if self.write_count % self.autosave_every == 0:
            self._flush_and_sync()

    def _finish_json(self):
        if not self._open:
            return
        if self.array_only:
            self.f.write('\n]\n')     # Close array
        else:
            self.f.write('\n]}\n')    # Close records/object
        self._flush_and_sync()
        self.f.close()
        self._open = False

    def _flush_and_sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def commit(self):
        if self._committed:
            return
        self._finish_json()
        try:
            os.replace(self.tmp_path, self.out_path)
            self._committed = True
        except Exception as e:
            print(f"commit() failed (temp file kept): {e}")

    def abort(self):
        self._finish_json()




def call_llm(inputs, sys_prompt, out_path, model,
             per_req_sleep=0.0, retries=2, show_progress=True,
             stop_on_error=True, max_workers=5):
    with open(sys_prompt, encoding="utf-8") as f:
        sp = f.read()

    meta = {
        "model": model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total_inputs": len(inputs),
        "notes": "Single-domain file. Each record is one (user, this domain)."
    }
    writer = JsonArrayWriter(out_path, meta, autosave_every=20, array_only=True)

    # Thread-local client (one OpenAI client per thread)
    _tls = threading.local()

    def _get_client():
        if getattr(_tls, "client", None) is None:
            _tls.client = OpenAI(api_key=OPENAI_API_KEY)
        return _tls.client

    def _one_request(i: int):
        obj = inputs[i]
        uid = obj.get("User ID")
        domain = obj.get("Domain")
        n_hist = len(obj.get("History", []))

        msgs = [
            {"role": "system", "content": sp},
            {"role": "user", "content": json.dumps(obj, ensure_ascii=False)}
        ]

        last_err = None
        result_obj = None
        t_head = time.time()

        for attempt in range(retries + 1):
            try:
                client = _get_client()
                res = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                content = res.choices[0].message.content
                result_obj = json.loads(content)  # raises on JSON parse failure
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(0.5 * (2 ** attempt))

        if result_obj is None:
            # Keep legacy failure record format
            result_obj = {
                "User ID": uid,
                "Domain": domain,
                "error": last_err,
                "raw_request": obj
            }
            ok = False
        else:
            ok = True

        if per_req_sleep > 0:
            time.sleep(per_req_sleep)

        elapsed = time.time() - t_head
        log = {"i": i, "uid": uid, "domain": domain, "n_hist": n_hist, "ok": ok, "elapsed": elapsed, "err": last_err}
        return i, result_obj, log

    completed = False
    try:
        n = len(inputs)
        # Results can arrive out of order; buffer and flush in index order.
        pending = {}
        next_to_write = 0

        if show_progress:
            pbar = tqdm(total=n, dynamic_ncols=True, desc="Persona generation", unit="user")
        else:
            pbar = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_one_request, i): i for i in range(n)}

            for fut in concurrent.futures.as_completed(futures):
                i, result_obj, log = fut.result()
                pending[i] = (result_obj, log)

                # Write in-order as much as possible
                while next_to_write in pending:
                    obj_to_write, log_to_print = pending.pop(next_to_write)

                    writer.write_obj(obj_to_write)

                    if show_progress:
                        if log_to_print["ok"]:
                            tqdm.write(f'{next_to_write+1}/{n} {log_to_print["domain"]}/{log_to_print["uid"]} items={log_to_print["n_hist"]} OK {log_to_print["elapsed"]:.2f}s')
                        else:
                            tqdm.write(f'{next_to_write+1}/{n} {log_to_print["domain"]}/{log_to_print["uid"]} items={log_to_print["n_hist"]} ERR(retries_exhausted)')
                            if stop_on_error:
                                # Policy: stop immediately on failure
                                raise RuntimeError(f'LLM failure: {log_to_print["err"]}')

                    next_to_write += 1
                    if pbar is not None:
                        pbar.update(1)

        completed = True

    except KeyboardInterrupt:
        print("User interrupt: progress kept in .tmp")
        writer.abort()
        return
    except Exception as e:
        print(f"Stopped: {e}\nProgress kept in .tmp -> {out_path}.tmp")
        writer.abort()
        return
    else:
        if completed:
            writer.commit()
            print(f"Done: committed to {out_path}")
    finally:
        if show_progress and 'pbar' in locals() and pbar is not None:
            pbar.close()


# Main
def main():
    ap = argparse.ArgumentParser(description="Generate personas for ALL overlapping users per domain (one JSON per domain).")
    ap.add_argument("--review_dir", required=True, help="{domain}.json per domain")
    ap.add_argument("--meta_dir", required=True, help="f_meta_{domain}.json directory")
    ap.add_argument("--domain_desc_dir", required=True, help="*_desc.json directory")
    ap.add_argument("--user_info_dir", required=True, help="*_usr_info.json directory")
    ap.add_argument("--system", required=True, help="System prompt path")
    ap.add_argument("--selected_users", help="Overlapping user list (f_usr_id.json)")
    ap.add_argument("--out_dir", required=True, help="Output directory per domain")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max_hist_per_user", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--tvt_root", required=True, help="train.txt root")
    ap.add_argument("--workers", type=int, default=5, help="Parallel LLM worker threads")
    args = ap.parse_args()

    # 1) Prepare domain context
    DOMAINS = ["Home_and_Kitchen", "Toys_and_Games"]

    print("[L] Loading meta/desc/user info...")
    meta = load_item_meta(args.meta_dir)
    domain_descs = load_dom_desc(args.domain_desc_dir)
    user_info = load_user_info(args.user_info_dir)

    print("[L] Loading maps/train per domain...")
    dom_ctx = {}
    dom_train_users_raw = {}
    for dom in DOMAINS:
        u2i, i2i, _, _, train_pairs, users_in_train_raw = _load_domain_maps_and_train(args.tvt_root, dom)
        dom_ctx[dom] = (u2i, i2i, train_pairs)
        dom_train_users_raw[dom] = users_in_train_raw
        print(f" - {dom}: TRAIN users = {len(users_in_train_raw):,}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 2) Run per domain (no sampling; use all users)
    for dom in DOMAINS:
        print(f"\n=== [{dom}] start ===")
        allowed = set(dom_train_users_raw[dom])   # use all users

        print(f"[L2] Review scan over TRAIN only (recent #{args.max_hist_per_user})...")
        u2i, i2i, train_pairs = dom_ctx[dom]
        reviews = scan_usr_rev_train_only(
            review_dir=args.review_dir,
            domain=dom,
            allowed_raw_users=allowed,          # all users for this domain
            max_hist_per_user=args.max_hist_per_user,
            user2idx=u2i,
            item2idx=i2i,
            train_pairs=train_pairs
        )

        print("[B] Building inputs...")
        inputs = build_inputs(
            dom,
            meta=meta,
            reviews=reviews,          # {dom: {uid_raw: [..]}}
            domain_descs=domain_descs,
            user_info=user_info
        )

        # Safety filter (just in case)
        inputs = [x for x in inputs if x.get("User ID") in allowed]
        print(f" - Target users: {len(inputs):,}")

        if not inputs:
            print(f" - {dom}: no inputs; skipping")
            continue

        out_path = os.path.join(args.out_dir, f"personas_{dom}.json")
        print(f"[RUN] LLM call -> {out_path}")

        call_llm(
            inputs=inputs,
            sys_prompt=args.system,
            out_path=out_path,
            model=args.model,
            per_req_sleep=args.sleep,
            retries=args.retries,
            show_progress=not args.no_progress
        )
        print(f"[DONE] {dom} -> {out_path}")

    print("\nAll domains completed.")


if __name__ == "__main__":
    main()
