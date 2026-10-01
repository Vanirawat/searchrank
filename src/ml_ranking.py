"""
ml_ranking.py - RandomForest ranking model and comparison with the weighted score.
"""
import sys

import joblib
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.db import load_products
from src.ranking import rank_products

LABELS_CSV = "data/processed/labels.csv"
MODEL_PATH = "data/rf_ranker.joblib"
FEATURES = ["similarity", "sim_n", "rating_n", "reviews_n", "price_n", "has_rating"]


def _model():
    return RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)


def evaluate(df: pd.DataFrame) -> None:
    """Hold out whole queries for testing (how well does it work on unseen queries?)."""
    gkf = GroupKFold(n_splits=min(4, df["query"].nunique()))
    rf_corr, base_corr = [], []

    for tr, te in gkf.split(df, groups=df["query"]):
        m = _model().fit(df.iloc[tr][FEATURES], df.iloc[tr]["label"])
        test = df.iloc[te].copy()
        test["pred"] = m.predict(test[FEATURES])
        for _, g in test.groupby("query"):
            if len(g) > 2 and g["label"].nunique() > 1:
                rf_corr.append(spearmanr(g["pred"], g["label"])[0])
                base_corr.append(spearmanr(g["score"], g["label"])[0])

    print("Ranking quality (Spearman correlation with manual labels, 1 = perfect match):")
    print(f"  Weighted score : {pd.Series(base_corr).mean():.3f}")
    print(f"  RandomForest   : {pd.Series(rf_corr).mean():.3f}")


def train() -> None:
    df = pd.read_csv(LABELS_CSV)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    print(f"Labeled rows: {len(df)} | Queries: {df['query'].nunique()}\n")

    evaluate(df)

    model = _model().fit(df[FEATURES], df["label"])
    joblib.dump(model, MODEL_PATH)
    print("\nFeature importance:")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(imp.round(3).to_string())
    print(f"\nModel saved: {MODEL_PATH}")


def rank_products_ml(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Rank with the ML model instead of the weighted formula."""
    model = joblib.load(MODEL_PATH)
    r = rank_products(df, query)
    if r.empty:
        return r
    r["ml_score"] = model.predict(r[FEATURES]).round(3)
    return r.sort_values("ml_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        res = rank_products_ml(load_products(query=q), q)
        print(res[["title", "price", "rating", "reviews", "ml_score", "score"]].head(10).to_string())
    else:
        train()