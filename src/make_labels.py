"""
make_labels.py - Har query se 10 random products nikalke labeling ke liye CSV banata hai.
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
        raise SystemExit(f"{OUT} pehle se hai. Delete mat karna agar labels bhar chuki ho, warna mehnat jayegi.")

    all_q = load_products()["query"].unique()
    parts = []
    for q in sorted(all_q):
        d = load_products(query=q)
        r = rank_products(d, q, min_similarity=0.0)   # kuch bhi filter nahi
        parts.append(r.sample(min(10, len(r)), random_state=42))

    out = pd.concat(parts, ignore_index=True)
    out["label"] = ""
    out[COLS].to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(out)} rows save hui: {OUT}")