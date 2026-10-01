"""
similarity.py - TF-IDF + cosine similarity se query-product relevance score.
"""
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.clean import clean_title
from src.db import load_products


def add_similarity(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """df mein 'similarity' column jodta hai aur usi hisaab se sort karta hai."""
    df = df.copy()
    q = clean_title(query)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(df["title_clean"].tolist() + [q])

    # Aakhri row query hai, baaki sab products
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    df["similarity"] = scores.round(4)

    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "running shoes"

    # Test 1: sirf us query ke products
    own = load_products(query=q)
    if own.empty:
        print(f"'{q}' ka data database mein nahi hai. collect_data ki QUERIES mein hona chahiye.")
        sys.exit()

    print(f"--- '{q}' ke products, similarity ke hisaab se ---")
    res = add_similarity(own, q)
    print(res[["title", "price", "similarity"]].head(10).to_string())

    # Test 2: saare products mein dhundo (galat category wale kitne upar aate hain?)
    allp = load_products()
    res_all = add_similarity(allp, q)
    top = res_all.head(15)
    print(f"\n--- Poore dataset mein top 15 (query: {q}) ---")
    print(top[["query", "title", "similarity"]].to_string())
    print(f"\nTop 15 mein sahi category ke: {(top['query'] == q).sum()}/15")