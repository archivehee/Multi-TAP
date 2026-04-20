import os
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

domain_pair = "Home_and_Kitchen, Toys_and_Games"  
t_usr_pair = "home_toys"
REVIEW_DIR = f"./data/amazon/{t_usr_pair}/filtered_data/f_usr_reviews"
META_DATA_DIR = f"./data/amazon/{t_usr_pair}/filtered_data/f_item_meta"
OUTPUT_DIR = f"./profiles/user_info/{t_usr_pair}"
USER_ID_PATH = f"{REVIEW_DIR}/f_usr_{t_usr_pair}.json"


def get_target_domains(meta_dir: str, pair_str: str) -> list[str]:
    all_domains = sorted(
        fname[len("f_meta_"):-5]
        for fname in os.listdir(meta_dir)
        if fname.startswith("f_meta_") and fname.endswith(".json")
    )
    if not pair_str:
        return all_domains
    requested = [dom.strip() for dom in pair_str.split(",") if dom.strip()]
    missing = [dom for dom in requested if dom not in all_domains]
    if missing:
        raise ValueError(f"Missing meta for domains: {missing}")
    return requested

def load_selected_users(path):
    with open(path, "r", encoding="utf-8") as f:
        return {obj.get("user_id") for obj in json.load(f) if obj.get("user_id")}


def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        if first == "[":
            f.seek(0)
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

def extract_cat(item):
    cats = item.get("categories", [])
    if isinstance(cats, list) and len(cats) > 1 and isinstance(cats[1], str):
        cat = cats[1].strip()
        return cat or None
    return None



def qcut_LMH_on_positive(series):

    labels = pd.Series(index=series.index, dtype="object")
    pos_mask = series > 0
    if pos_mask.sum() < 3:
        return labels

    ranks = series[pos_mask].rank(method="average", pct=True)  # 0~1
    buckets = np.ceil(ranks * 3).clip(1, 3).astype(int)
    map_lmh = {1: "L", 2: "M", 3: "H"}
    labels.loc[pos_mask] = buckets.map(map_lmh)
    return labels


def per_category_LMH_from_user_values(uc_map):
    cat_to_series = defaultdict(lambda: pd.Series(dtype=float))
    for (u, c), v in uc_map.items():
        cat_to_series[c].loc[u] = float(v)

    out = defaultdict(dict)
    for cat, s in cat_to_series.items():
        labels = qcut_LMH_on_positive(s)
        for uid, tag in labels.dropna().items():
            out[uid][cat] = str(tag)
    return out


def compute_bins(cat_to_values):
    bins = {}
    for cat, values in cat_to_values.items():
        if len(values) < 3:
            continue
        s = pd.Series(values, dtype=float)
        q1, q2 = s.quantile([0.33, 0.66]).tolist()
        bins[cat] = (float(q1), float(q2))
    return bins


def group_users_by_tertiles(uc_scores):
    cat_user_scores = defaultdict(dict)
    for (u, c), v in uc_scores.items():
        cat_user_scores[c][u] = float(v)

    out = defaultdict(dict)
    for cat, user_scores in cat_user_scores.items():
        if len(user_scores) < 3:
            continue
        s = pd.Series(user_scores)
        q_low, q_high = s.quantile([0.33, 0.66]).tolist()
        for u, score in user_scores.items():
            if score >= q_high:
                group = "H"
            elif score >= q_low:
                group = "M"
            else:
                group = "L"
            out[u][cat] = group
    return out


def diversity_LMH_from_counts(user_to_count):

    s = pd.Series(user_to_count, dtype=float)
    labels = qcut_LMH_on_positive(s)  
    return {u: (str(lbl) if pd.notna(lbl) else "Unknown") for u, lbl in labels.items()} 



def build_domain_user_info(domain, selected_users):
    meta_path = os.path.join(META_DATA_DIR, f"f_meta_{domain}.json")
    review_path = os.path.join(REVIEW_DIR, f"{domain}.json")

    asin2cat, asin2price, asin2avg, asin2ratnum = {}, {}, {}, {}
    for it in iter_records(meta_path):
        asin = it.get("parent_asin") or it.get("asin")
        if not asin:
            continue
        cat = extract_cat(it)
        if not cat:
            continue
        asin2cat[asin] = cat

        price = it.get("price")
        if isinstance(price, (int, float)):
            asin2price[asin] = float(price)
        avg_rating = it.get("average_rating")
        if isinstance(avg_rating, (int, float)):
            asin2avg[asin] = float(avg_rating)
        rating_number = it.get("rating_number")
        if isinstance(rating_number, (int, float)):
            asin2ratnum[asin] = float(rating_number)

    interactions = []  # (user, asin)
    for r in iter_records(review_path):
        uid = r.get("user_id")
        if not uid or uid not in selected_users:
            continue
        asin = r.get("parent_asin") or r.get("asin")
        if not asin or asin not in asin2cat:
            continue
        interactions.append((uid, asin))

    if not interactions:
        return {}

    cat_to_prices = defaultdict(list)
    cat_to_ratings = defaultdict(list)
    cat_to_ratnums = defaultdict(list)
    for asin, cat in asin2cat.items():
        if asin in asin2price:
            cat_to_prices[cat].append(asin2price[asin])
        if asin in asin2avg:
            cat_to_ratings[cat].append(asin2avg[asin])
        if asin in asin2ratnum:
            cat_to_ratnums[cat].append(asin2ratnum[asin])

    cat_price_bins = compute_bins(cat_to_prices)
    cat_rating_bins = compute_bins(cat_to_ratings)
    cat_ratnum_bins = compute_bins(cat_to_ratnums)

    price_tags = defaultdict(list)   # (u,c) -> [1/2/3]
    rating_tags = defaultdict(list)  # (u,c) -> [1/2/3]
    ratnum_tags = defaultdict(list)  # (u,c) -> [1/2/3]
    cat_freq = defaultdict(Counter)  # u -> Counter(cat)

    for uid, asin in interactions:
        cat = asin2cat.get(asin)
        if not cat:
            continue
        cat_freq[uid][cat] += 1

        if asin in asin2price and cat in cat_price_bins:
            p = asin2price[asin]
            q1, q2 = cat_price_bins[cat]
            tag = 1 if p < q1 else 2 if p < q2 else 3
            price_tags[(uid, cat)].append(tag)

        if asin in asin2avg and cat in cat_rating_bins:
            r = asin2avg[asin]
            q1, q2 = cat_rating_bins[cat]
            tag = 1 if r < q1 else 2 if r < q2 else 3
            rating_tags[(uid, cat)].append(tag)

        if asin in asin2ratnum and cat in cat_ratnum_bins:
            n = asin2ratnum[asin]
            q1, q2 = cat_ratnum_bins[cat]
            tag = 1 if n < q1 else 2 if n < q2 else 3
            ratnum_tags[(uid, cat)].append(tag)

    uc_price_avg = {k: float(np.mean(v)) for k, v in price_tags.items() if v}
    uc_rating_avg = {k: float(np.mean(v)) for k, v in rating_tags.items() if v}
    uc_ratnum_avg = {k: float(np.mean(v)) for k, v in ratnum_tags.items() if v}

    user_price_LMH = group_users_by_tertiles(uc_price_avg)
    user_rating_LMH = group_users_by_tertiles(uc_rating_avg)
    user_ratnum_LMH = group_users_by_tertiles(uc_ratnum_avg)


    all_users = sorted(selected_users)
    all_cats = sorted({c for _, c in uc_price_avg.keys()} |
                      {c for _, c in uc_rating_avg.keys()} |
                      {c for _, c in uc_ratnum_avg.keys()} |
                      {c for u in cat_freq for c in cat_freq[u].keys()})

    data = {cat: pd.Series({u: cat_freq[u][cat] for u in all_users}, dtype=float) for cat in all_cats}
    user_cat_df = pd.DataFrame(data, index=all_users).fillna(0)

    cats_fam_labels = defaultdict(dict)  
    for cat in all_cats:
        lmhs = qcut_LMH_on_positive(user_cat_df[cat])
        for u, lab in lmhs.dropna().items():
            cats_fam_labels[u][cat] = str(lab)

    user_distinct_cat_counts = {u: int((user_cat_df.loc[u].values > 0).sum()) for u in all_users}
    user_diversity_LMH = diversity_LMH_from_counts(user_distinct_cat_counts)

    out = []
    for uid in sorted(selected_users):
        out.append({
            "user_id": uid,
            "price_affiliated_group": user_price_LMH.get(uid, {}),
            "rating_score_preferred_group": user_rating_LMH.get(uid, {}),
            "rating_nums_preferred_group": user_ratnum_LMH.get(uid, {}),
            "cats_familiarity": cats_fam_labels.get(uid, {}),
            "cats_interaction_diversity": user_diversity_LMH.get(uid, "L"),
        })
    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected_users = load_selected_users(USER_ID_PATH)
    print(f"# selected users: {len(selected_users):,}")

    domains = get_target_domains(META_DATA_DIR, domain_pair)
    print("Domains:", domains)

    for domain in domains:
        print(f"\n[Domain] {domain} — building user info ...")
        result = build_domain_user_info(domain, selected_users)
        out_path = os.path.join(OUTPUT_DIR, f"{domain}_usr_info.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path} (users: {len(result):,})")


if __name__ == "__main__":
    main()
