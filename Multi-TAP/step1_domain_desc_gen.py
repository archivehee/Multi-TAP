import os
import re
import json
import random
import math
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from openai import OpenAI
from config import OPENAI_API_KEY


PROJECT_ROOT = Path(__file__).resolve().parent
target_pair = "home_toys"
DATA_ROOT = PROJECT_ROOT/"data"/"amazon"/target_pair /"filtered_data"
META_DATA_DIR = str(DATA_ROOT/"f_item_meta")
INTERACTION_DATA_DIR = str(DATA_ROOT/"f_usr_reviews")
SYSTEM_PROMPT_PATH = str(PROJECT_ROOT/ "system_prompt"/"domain_report_gen.txt")
OUTPUT_DIR = str(PROJECT_ROOT/"profiles"/"domain_desc")


DOMAINS = [
    "Home_and_Kitchen",
    "Toys_and_Games"
]

RANDOM_SEED = 2025
SAMPLE_SIZE_PER_DOMAIN = 100
SPLIT_INTERACTION = 0.5   # 50%
SPLIT_RATING = 0.5        # 50%

MAX_INPUT_TOKENS = 20000
PER_ITEM_TOKEN_CAP = 275
PER_ITEM_TEXT_CHAR_CAP = 280
SAFETY_OVERHEAD_TOKENS = 1500


def _clean_text(s):
    if not s:
        return ""
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(Review|About the Author|Praise for)\b.*?$", "", s, flags=re.I)
    return s

def _truncate(s, max_len):
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    last_period = cut.rfind(".")
    if last_period >= int(max_len * 0.6):
        return cut[:last_period + 1]
    return cut + "..."

def _estimate_tokens(s):
    return max(1, len(s) // 4)


def _normalize_domain_name(name: str) -> str:
    return str(name).strip()

def _norm_category(cat):
    if isinstance(cat, list) and cat:
        return str(cat[-1])
    if isinstance(cat, str) and cat.strip():
        return cat.strip()
    return "Unknown"

def _extract_cat(item):
    categories = item.get("categories", [])
    if isinstance(categories, list) and len(categories) > 1 and isinstance(categories[1], str):
        cat = categories[1].strip()
        if cat:
            return cat
    cat = item.get("category")
    if isinstance(cat, str) and cat.strip():
        return cat.strip()
    return "Unknown"

def _asin_of(item):
    return item.get("parent_asin") or item.get("asin")


def load_meta(meta_dir, domains=None):
    out = {}
    domain_set = set(_normalize_domain_name(d) for d in (domains or []))

    for file in os.listdir(meta_dir):
        if not file.endswith(".json"):
            continue

        domain = file[:-5]
        if domain.startswith("f_meta_"):
            domain = domain[len("f_meta_"):]
        domain = _normalize_domain_name(domain)

        # Filter: only keep requested domains
        if domain_set and domain not in domain_set:
            continue

        meta_path = os.path.join(meta_dir, file)

        items = []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                head = f.read(1)
                f.seek(0)
                if head == "[":
                    items = json.load(f)
        except Exception:
            items = []

        m = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            asin = _asin_of(item)
            if not asin:
                continue
            item["category"] = _extract_cat(item)
            m[asin] = item

        out[domain] = m

    return out


def load_interaction_counts_for_domain(domain, interactions_dir):
 
    counts = Counter()
    if not interactions_dir or not os.path.isdir(interactions_dir):
        return counts

    domain = _normalize_domain_name(domain)

    # try "<domain>.json"
    path = os.path.join(interactions_dir, f"{domain}.json")
    if not os.path.exists(path):
        # try alternative: "f_usr_reviews_<domain>.json" (robustness)
        alt = os.path.join(interactions_dir, f"f_usr_reviews_{domain}.json")
        path = alt if os.path.exists(alt) else None

    if not path:
        return counts

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for r in data:
                if not isinstance(r, dict):
                    continue
                asin = r.get("parent_asin") or r.get("asin")
                if asin:
                    counts[asin] += 1
    except Exception:
        pass

    return counts


def _rating_from_meta(m):
    v = m.get("average_rating")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None  # treat as missing

def category_shares_by_interaction(meta_map, inter_counts):
    by_cat = Counter()
    for asin, m in meta_map.items():
        cat = _norm_category(m.get("category"))
        by_cat[cat] += inter_counts.get(asin, 0)

    total_inter = sum(by_cat.values())
    if total_inter > 0:
        return {c: n / total_inter for c, n in by_cat.items()}

    # fallback: inventory
    inv = Counter(_norm_category(m.get("category")) for m in meta_map.values())
    total = sum(inv.values())
    return {c: n / total for c, n in inv.items()} if total else {"Unknown": 1.0}

def _largest_remainder_alloc(shares, quota):
    exact = {k: shares[k] * quota for k in shares}
    base = {k: int(math.floor(exact[k])) for k in shares}
    remaining = quota - sum(base.values())
    order = sorted(((k, exact[k] - base[k]) for k in shares), key=lambda x: (-x[1], x[0]))
    for i in range(max(0, remaining)):
        base[order[i % len(order)][0]] += 1
    return base


def stratified_two_perspective_sample(meta_map, inter_counts, total_n, split_inter=0.5, split_rating=0.5, seed=42):

    random.seed(seed)

    # Organize pools by category
    by_cat = defaultdict(list)
    for asin, m in meta_map.items():
        by_cat[_norm_category(m.get("category"))].append(asin)

    shares = category_shares_by_interaction(meta_map, inter_counts)
    alloc_cat = _largest_remainder_alloc(shares, total_n)

    # Precompute ranking metrics
    interaction_rank = defaultdict(int, inter_counts)
    rating_rank = {}
    for asin, m in meta_map.items():
        r = _rating_from_meta(m)
        rating_rank[asin] = (r if r is not None else -1.0)

    selected = []
    selected_set = set()
    spill_inter = defaultdict(list)
    spill_rating = defaultdict(list)

    # Per-category selection
    for cat, alloc in alloc_cat.items():
        pool = by_cat.get(cat, [])
        if not pool:
            continue

        n_inter = int(round(alloc * split_inter))
        n_rate = max(0, alloc - n_inter)

        # Sort pools
        pool_inter_sorted = sorted(
            pool,
            key=lambda a: (interaction_rank.get(a, 0), rating_rank.get(a, -1.0)),
            reverse=True
        )
        pool_rate_sorted = sorted(
            pool,
            key=lambda a: (rating_rank.get(a, -1.0), interaction_rank.get(a, 0)),
            reverse=True
        )

        # Pick by interaction
        for a in pool_inter_sorted:
            if len(selected) >= total_n:
                break
            if len([x for x in selected if _norm_category(meta_map.get(x, {}).get("category")) == cat]) >= n_inter:
                break
            if a in selected_set:
                continue
            selected.append(a)
            selected_set.add(a)

        spill_inter[cat] = [a for a in pool_inter_sorted if a not in selected_set]

        # Pick by rating
        picked_rate = 0
        for a in pool_rate_sorted:
            if picked_rate >= n_rate:
                break
            if a in selected_set:
                continue
            selected.append(a)
            selected_set.add(a)
            picked_rate += 1

        spill_rating[cat] = [a for a in pool_rate_sorted if a not in selected_set]

    # If still short, backfill: same-category spills first (inter then rating), then cross-category
    def _drain_spills(spills):
        nonlocal selected, selected_set
        for cat in spills:
            for a in spills[cat]:
                if len(selected) >= total_n:
                    return
                if a in selected_set:
                    continue
                selected.append(a)
                selected_set.add(a)

    if len(selected) < total_n:
        _drain_spills(spill_inter)
    if len(selected) < total_n:
        _drain_spills(spill_rating)

    # Cross-category global backfill: build a unified ranking (interaction then rating)
    if len(selected) < total_n:
        all_asins = list(meta_map.keys())
        global_sorted = sorted(
            all_asins,
            key=lambda a: (interaction_rank.get(a, 0), rating_rank.get(a, -1.0)),
            reverse=True
        )
        for a in global_sorted:
            if len(selected) >= total_n:
                break
            if a in selected_set:
                continue
            selected.append(a)
            selected_set.add(a)

    return selected[:total_n]


def build_item_text(meta, max_desc_len=PER_ITEM_TEXT_CHAR_CAP):
    title = _clean_text(str(meta.get("title") or "").strip())
    category = _extract_cat(meta)

    desc = meta.get("description")
    if isinstance(desc, list):
        desc = " ".join([str(x) for x in desc if x])
    if not desc:
        feats = meta.get("features") or meta.get("feature") or meta.get("bullet_points")
        if isinstance(feats, list):
            desc = " ".join([str(x) for x in feats if x])
        elif isinstance(feats, str):
            desc = feats
    desc = _clean_text(desc or "")
    desc = _truncate(desc, max_desc_len)

    parts = [p for p in [title, category, desc] if p]
    text = " || ".join(parts)

    if _estimate_tokens(text) > PER_ITEM_TOKEN_CAP:
        text = _truncate(text, PER_ITEM_TOKEN_CAP * 4)
    return text

def _normalize_keywords(obj):
    kw = (
        obj.get("Domain Profile", {}).get("Domain keywords")
        or obj.get("domain_profile", {}).get("domain_keywords")
        or obj.get("domain_keywords")
        or []
    )
    if isinstance(kw, str):
        kw = [t.strip().lower() for t in kw.split(",") if t.strip()]
    elif isinstance(kw, list):
        kw = [str(t).strip().lower() for t in kw if str(t).strip()]
    else:
        kw = []
    seen = set()
    deduped = []
    for t in kw:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:10]


def build_llm_input_for_domain(domain, meta_map, sampled_asins):
    sampled = {}
    budget_tokens = MAX_INPUT_TOKENS - SAFETY_OVERHEAD_TOKENS
    used_tokens = _estimate_tokens(domain) + 50

    for asin in sampled_asins:
        m = meta_map.get(asin, {})
        text = build_item_text(m)
        need = _estimate_tokens(text) + 30
        if used_tokens + need > budget_tokens:
            break
        sampled[asin] = text
        used_tokens += need

    return {"Domain": domain, "sampled_item_list": sampled}


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20").strip()

def call_llm(client, messages, temperature=1.0):
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content

def run_llm(domain_inputs, system_prompt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = OpenAI(api_key=OPENAI_API_KEY)

    for obj in tqdm(domain_inputs, desc="Generating domain descriptions"):
        domain = obj["Domain"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(obj, ensure_ascii=False)},
        ]
        content = call_llm(client, messages)

        try:
            out_json = json.loads(content)
        except Exception:
            out_json = {"Domain": domain, "raw_response": content}

        save_path = os.path.join(output_dir, f"{domain}_desc.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

        # Console preview
        desc = (
            out_json.get("Domain Profile", {}).get("Domain Description")
            or out_json.get("domain_profile", {}).get("domain_description")
            or out_json.get("raw_response", "")
        )
        keywords = _normalize_keywords(out_json)

        print(f"\n[{domain}] model: {OPENAI_MODEL}")
        print(f"[{domain}] Domain Description:\n{desc}\n")
        if keywords:
            print(f"[{domain}] Domain keywords (10): {', '.join(keywords)}\n")


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Normalize domain list once
    domain_list = [_normalize_domain_name(d) for d in DOMAINS]

    print("[1/5] Loading item metadata (filtered domains only)...")
    meta_per_domain = load_meta(META_DATA_DIR, domains=domain_list)
    print(f" - #Domains loaded: {len(meta_per_domain)}")
    print(f" - Domains: {list(meta_per_domain.keys())}")

    inputs = []

    print("[2/5] Loading interaction counts...")
    print("[3/5] Sampling per domain with (interaction 50% + rating 50%) within category shares...")

    # Preserve your domain order: iterate in DOMAINS order, but only if meta exists
    for domain in domain_list:
        meta_map = meta_per_domain.get(domain)
        if not meta_map:
            print(f" - Skip: {domain} (no meta file found in META_DATA_DIR)")
            continue

        inter_counts = load_interaction_counts_for_domain(domain, INTERACTION_DATA_DIR)

        sampled_asins = stratified_two_perspective_sample(
            meta_map=meta_map,
            inter_counts=inter_counts,
            total_n=SAMPLE_SIZE_PER_DOMAIN,
            split_inter=SPLIT_INTERACTION,
            split_rating=SPLIT_RATING,
            seed=RANDOM_SEED,
        )

        inputs.append(build_llm_input_for_domain(domain, meta_map, sampled_asins))

    print("[4/5] Calling LLM...")
    run_llm(inputs, SYSTEM_PROMPT_PATH, OUTPUT_DIR)
    print("[5/5] Done. Saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
