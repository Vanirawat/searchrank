"""
compare.py - Same product ki alag-alag sellers ki listings ko group karke price compare karna.
"""
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def group_products(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Milte-julte titles ko ek group_id deta hai (title similarity se)."""
    df = df.reset_index(drop=True).copy()
    if df.empty:
        df["group_id"] = []
        return df

    matrix = TfidfVectorizer(stop_words="english").fit_transform(df["title_clean"])
    sims = cosine_similarity(matrix)

    group, reps = [], []          # reps = har group ka pehla (sabse upar rank wala) product
    for i in range(len(df)):
        for g, r in enumerate(reps):
            if sims[i, r] >= threshold:
                group.append(g)
                break
        else:
            group.append(len(reps))
            reps.append(i)

    df["group_id"] = group
    return df


def price_comparison(ranked: pd.DataFrame, threshold: float = 0.5,
                     top_n: int = 30, min_offers: int = 1) -> pd.DataFrame:
    """Ranked products ko group karke har product ka price summary banata hai."""
    g = group_products(ranked.head(top_n), threshold)
    rows = []
    for _, grp in g.groupby("group_id", sort=False):
        cheapest = grp.loc[grp["price"].idxmin()]
        rows.append({
            "product": grp.iloc[0]["title"],
            "offers": len(grp),
            "sellers": ", ".join(sorted(grp["seller"].dropna().astype(str).unique())),
            "lowest_price": grp["price"].min(),
            "highest_price": grp["price"].max(),
            "you_save": round(grp["price"].max() - grp["price"].min(), 2),
            "cheapest_seller": cheapest["seller"],
            "link": cheapest["link"],
        })
    out = pd.DataFrame(rows)
    return out[out["offers"] >= min_offers].reset_index(drop=True)


if __name__ == "__main__":
    from src.db import load_products
    from src.ranking import rank_products

    q = " ".join(sys.argv[1:]) or "air fryer"
    ranked = rank_products(load_products(query=q), q)
    res = price_comparison(ranked, min_offers=2)
    print(res[["product", "offers", "lowest_price", "highest_price", "you_save", "cheapest_seller"]].to_string())