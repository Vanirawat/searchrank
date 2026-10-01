"""
similarity.py - Query-product relevance score using TF-IDF + cosine similarity.
"""
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.clean import clean_title
from src.db import load_products


def add_similarity(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Add a 'similarity' column (0 to 1) and sort by it."""
    df = df.copy()
    q = clean_title(query)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(df["title_clean"].tolist() + [q])

    # Last row is the query, the rest are products
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    df["similarity"] = scores.round(4)

    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "running shoes"

    # Test 1: only products of this query
    own = load_products(query=q)
    if own.empty:
        print(f"No data for '{q}' in the database. It must be in QUERIES in collect_data.py.")
        sys.exit()

    print(f"--- Products for '{q}', sorted by similarity ---")
    res = add_similarity(own, q)
    print(res[["title", "price", "similarity"]].head(10).to_string())

    # Test 2: search across the whole dataset (do wrong-category products rank high?)
    allp = load_products()
    res_all = add_similarity(allp, q)
    top = res_all.head(15)
    print(f"\n--- Top 15 across the whole dataset (query: {q}) ---")
    print(top[["query", "title", "similarity"]].to_string())
    print(f"\nCorrect category in top 15: {(top['query'] == q).sum()}/15")