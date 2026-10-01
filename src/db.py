"""
db.py - Clean products ko SQLite database mein save/load karna.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/searchrank.db")
CLEAN_CSV = "data/processed/products_clean.csv"


def save_products(df: pd.DataFrame, table: str = "products") -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def load_products(query: str | None = None, table: str = "products") -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        if query:
            sql = f"SELECT * FROM {table} WHERE query = ?"
            return pd.read_sql_query(sql, conn, params=(query,))
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV)
    save_products(df)
    print(f"{len(df)} products database mein save hue: {DB_PATH}")

    check = load_products()
    print(f"Database se wapas padhe: {len(check)} rows")
    print(check.groupby("query").size())