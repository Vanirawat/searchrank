"""
make_labels.py - Pick 10 random products per query and export a CSV for manual labeling.
"""
from pathlib import Path

import pandas as pd

from src.db import load_products
from src.ranking import rank_products

OUT = Path("data/processed/labels.csv")

COLS = ["query", "title", "price", "rating", "reviews", "seller", "label",
        "similarity", "sim_n", "rating_n", "reviews_n", "price_n", "has_rating", "score"]

if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists. Do not delete it if you have filled in labels.")

    all_q = load_products()["query"].unique()
    parts = []
    for q in sorted(all_q):
        d = load_products(query=q)
        r = rank_products(d, q, min_similarity=0.0)   # no filtering at all
        parts.append(r.sample(min(10, len(r)), random_state=42))

    out = pd.concat(parts, ignore_index=True)
    out["label"] = ""
    out[COLS].to_csv(OUT, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows: {OUT}")