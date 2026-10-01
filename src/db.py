"""
db.py - Save and load clean products in a SQLite database.
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


def upsert_query(df: pd.DataFrame, query: str, table: str = "products") -> None:
    """Replace the rows of one query without touching the rest of the table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            conn.execute(f"DELETE FROM {table} WHERE query = ?", (query,))
        except sqlite3.OperationalError:
            pass  # table does not exist yet
        df.to_sql(table, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_CSV)
    save_products(df)
    print(f"Saved {len(df)} products to database: {DB_PATH}")

    check = load_products()
    print(f"Read back from database: {len(check)} rows")
    print(check.groupby("query").size())