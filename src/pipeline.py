"""
pipeline.py - Any query: fetch -> clean -> save -> rank -> picks -> price comparison.
The Streamlit UI calls this.
"""
import sys
from pathlib import Path

import pandas as pd

from src.clean import clean_products
from src.compare import price_comparison
from src.db import upsert_query
from src.fetch import fetch_products
from src.ml_ranking import MODEL_PATH, rank_products_ml
from src.picks import get_picks
from src.ranking import rank_products


def run_search(query: str, use_ml: bool = True, refresh: bool = False) -> dict:
    """
    New query: live data from SerpApi (uses 1 of your searches).
    Previously searched query: loaded from cache (no API call).
    """
    query = " ".join(query.lower().split())
    raw = fetch_products(query, use_cache=not refresh)
    if raw.empty:
        return {"query": query, "ranked": pd.DataFrame(), "picks": {}, "comparison": pd.DataFrame()}

    clean = clean_products(raw)
    upsert_query(clean, query)

    if use_ml and Path(MODEL_PATH).exists():
        ranked = rank_products_ml(clean, query)
    else:
        ranked = rank_products(clean, query)

    return {
        "query": query,
        "ranked": ranked,
        "picks": get_picks(ranked),
        "comparison": price_comparison(ranked, min_offers=1),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "running shoes"
    out = run_search(q)

    print(f"\n=== Top 10: '{out['query']}' ===")
    print(out["ranked"][["title", "price", "rating", "reviews"]].head(10).to_string())

    print("\n=== Picks ===")
    for label, row in out["picks"].items():
        print(f"{label}: {row['title'][:60]} | Rs {row['price']} | rating {row['rating']}")

    print("\n=== Price comparison (products with 2+ sellers) ===")
    cmp_df = out["comparison"]
    print(cmp_df[cmp_df["offers"] >= 2][["product", "offers", "lowest_price", "highest_price", "you_save"]].to_string())