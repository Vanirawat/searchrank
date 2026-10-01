"""
clean.py - Clean the raw products CSV.
"""
import re

import pandas as pd

RAW_CSV = "data/processed/products_raw.csv"
CLEAN_CSV = "data/processed/products_clean.csv"


def clean_title(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows missing essential columns
    df = df.dropna(subset=["title", "price"])

    # Convert numeric columns to proper types
    for col in ["price", "rating", "reviews"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Missing rating/reviews = 0, and keep a flag
    df["has_rating"] = df["rating"].notna().astype(int)
    df["rating"] = df["rating"].fillna(0)
    df["reviews"] = df["reviews"].fillna(0)

    # Remove duplicates
    df = df.drop_duplicates(subset=["query", "title", "seller"])

    # Clean title for NLP
    df["title_clean"] = df["title"].apply(clean_title)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV)
    clean = clean_products(raw)
    clean.to_csv(CLEAN_CSV, index=False)

    print(f"Raw rows: {len(raw)} -> Clean rows: {len(clean)}")
    print(f"Products with a rating: {clean['has_rating'].sum()}")
    print("\nPrice summary:")
    print(clean["price"].describe().round(1))