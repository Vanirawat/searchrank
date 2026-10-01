"""
fetch.py - Fetch live product data from SerpApi (Google Shopping).
"""
import os
import re
import json
import sys
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search.json"
RAW_DIR = Path("data/raw")


def _cache_path(query: str) -> Path:
    """Turn a query into a file name, e.g. 'running shoes' -> running_shoes.json"""
    slug = re.sub(r"[^a-z0-9]+", "_", query.strip().lower()).strip("_")
    return RAW_DIR / f"{slug}.json"


def _call_api(query: str) -> dict:
    if not API_KEY:
        raise RuntimeError("SERPAPI_KEY not found. Check your .env file.")

    params = {
        "engine": "google_shopping",
        "q": query,
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",
        "api_key": API_KEY,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"SerpApi error: {data['error']}")
    return data


def fetch_products(query: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch products for a query and return them as a DataFrame.
    If use_cache is True, a previously saved response is reused (saves an API search).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(query)

    if use_cache and cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"[cache] Loaded '{query}' from saved file")
    else:
        data = _call_api(query)
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[api] Fetched '{query}' from SerpApi and saved it")

    rows = []
    for item in data.get("shopping_results", []):
        rows.append({
            "query": query,
            "title": item.get("title"),
            "price_text": item.get("price"),
            "price": item.get("extracted_price"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
            "seller": item.get("source"),
            "link": item.get("product_link") or item.get("link"),
            "thumbnail": item.get("thumbnail"),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "running shoes"
    df = fetch_products(q)
    print(f"\nTotal products: {len(df)}\n")
    print(df[["title", "price", "rating", "reviews", "seller"]].head(10).to_string())