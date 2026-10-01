"""
collect_data.py - Fetch data for many queries and save it as one CSV.
"""
import pandas as pd

from src.fetch import fetch_products

QUERIES = [
    "running shoes",
    "iphone 15",
    "samsung galaxy phone",
    "laptop under 50000",
    "wireless earbuds",
    "smart watch",
    "backpack",
    "bluetooth speaker",
    "office chair",
    "water bottle",
    "men tshirt",
    "air fryer",
]

if __name__ == "__main__":
    frames = []
    for q in QUERIES:
        try:
            frames.append(fetch_products(q))
        except Exception as e:
            print(f"[error] '{q}' failed: {e}")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv("data/processed/products_raw.csv", index=False)
    print(f"\nTotal rows: {len(df)} | Queries: {df['query'].nunique()}")