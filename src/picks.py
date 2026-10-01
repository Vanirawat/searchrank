"""
picks.py - Best Overall, Lowest Price, Highest Rated, Most Reviewed.
"""
import sys

import pandas as pd

from src.db import load_products
from src.ranking import rank_products


def get_picks(ranked: pd.DataFrame, min_reviews: int = 10) -> dict:
    if ranked.empty:
        return {}

    # Highest Rated: pehle woh jinke kam se kam min_reviews hon (1 review wala 5.0 fake hota hai)
    pool = ranked[ranked["reviews"] >= min_reviews]
    if pool.empty:
        pool = ranked[ranked["has_rating"] == 1]
    if pool.empty:
        pool = ranked

    return {
        "Best Overall": ranked.iloc[0],
        "Lowest Price": ranked.loc[ranked["price"].idxmin()],
        "Highest Rated": pool.sort_values(["rating", "reviews"], ascending=False).iloc[0],
        "Most Reviewed": ranked.loc[ranked["reviews"].idxmax()],
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "running shoes"
    data = load_products(query=q)
    if data.empty:
        print(f"'{q}' ka data database mein nahi hai.")
        sys.exit()

    ranked = rank_products(data, q)
    print(f"--- Top 10 ranked: '{q}' ---")
    print(ranked[["title", "price", "rating", "reviews", "similarity", "score"]].head(10).to_string())

    print("\n--- Picks ---")
    for label, row in get_picks(ranked).items():
        print(f"{label}: {row['title'][:60]} | Rs {row['price']} | rating {row['rating']} | reviews {int(row['reviews'])}")