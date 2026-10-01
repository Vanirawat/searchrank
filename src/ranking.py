"""
ranking.py - Weighted score se products ko rank karna.

score = 0.40*similarity + 0.25*rating + 0.20*reviews + 0.15*price
(price jitna kam, utna accha score)
"""
import numpy as np
import pandas as pd

from src.similarity import add_similarity

WEIGHTS = {"similarity": 0.40, "rating": 0.25, "reviews": 0.20, "price": 0.15}


def _minmax(s: pd.Series) -> pd.Series:
    """Values ko 0-1 range mein laata hai. Sab same ho toh 0.5."""
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def rank_products(df: pd.DataFrame, query: str,
                  weights: dict = WEIGHTS,
                  min_similarity: float = 0.05) -> pd.DataFrame:
    """Similarity jodta hai, irrelevant products hatata hai, score nikalke sort karta hai."""
    df = add_similarity(df, query)
    df = df[df["similarity"] >= min_similarity].copy()
    if df.empty:
        return df

    df["sim_n"] = _minmax(df["similarity"])
    df["rating_n"] = (df["rating"] / 5).clip(0, 1)
    df["reviews_n"] = _minmax(np.log1p(df["reviews"]))   # log, taaki bade numbers dominate na karein
    df["price_n"] = 1 - _minmax(df["price"])             # sasta = zyada score

    df["score"] = (
        weights["similarity"] * df["sim_n"]
        + weights["rating"] * df["rating_n"]
        + weights["reviews"] * df["reviews_n"]
        + weights["price"] * df["price_n"]
    ).round(4)

    return df.sort_values("score", ascending=False).reset_index(drop=True)