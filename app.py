"""
app.py - SearchRank Streamlit UI
Run: streamlit run app.py
"""
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from src.compare import price_comparison
from src.picks import get_picks
from src.pipeline import run_search

st.set_page_config(page_title="SearchRank", page_icon="🔎", layout="wide")
st.title("🔎 SearchRank")
st.caption("AI-powered e-commerce product search, ranking & price comparison")

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")
    use_ml = st.toggle("ML ranking (RandomForest)", value=True)
    refresh = st.checkbox("Fetch fresh data", value=False)
    st.caption("A new query or fresh data uses SerpApi search. Previously searched queries load from cache.")

# ---------- Search box ----------
with st.form("search_form"):
    query = st.text_input("What are you looking for?", placeholder="e.g. running shoes, air fryer, gaming mouse")
    submitted = st.form_submit_button("Search", type="primary")

if submitted and query.strip():
    with st.spinner("Searching and ranking products..."):
        try:
            st.session_state["result"] = run_search(query, use_ml=use_ml, refresh=refresh)
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Search failed: {e}")

result = st.session_state.get("result")
if result is None:
    st.info("Enter a product above and click Search.")
    st.stop()

ranked = result["ranked"]
if ranked.empty:
    st.warning("No products found for this query. Try a different one.")
    st.stop()

# ---------- Sidebar: filters ----------
pmin, pmax = float(ranked["price"].min()), float(ranked["price"].max())
with st.sidebar:
    st.header("Filters")
    if pmin < pmax:
        price_range = st.slider("Price range (₹)", pmin, pmax, (pmin, pmax))
    else:
        price_range = (pmin, pmax)
    min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5)

filtered = ranked[
    ranked["price"].between(price_range[0], price_range[1]) & (ranked["rating"] >= min_rating)
].reset_index(drop=True)

if filtered.empty:
    st.warning("No products match the current filters. Try relaxing them.")
    st.stop()

score_col = "ml_score" if "ml_score" in filtered.columns else "score"
st.subheader(f"Results for “{result['query']}”  ·  {len(filtered)} products")

# ---------- 4 Picks ----------
picks = get_picks(filtered)
cols = st.columns(4)
for col, (label, row) in zip(cols, picks.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            try:
                if pd.notna(row.get("thumbnail")):
                    st.image(row["thumbnail"], width=140)
            except Exception:
                pass
            st.caption(str(row["title"])[:70])
            st.markdown(f"### ₹{row['price']:,.0f}")
            st.caption(f"⭐ {row['rating']}  ·  {int(row['reviews'])} reviews  ·  {row['seller']}")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Ranked results", "Price comparison", "Charts"])

with tab1:
    show = filtered[["thumbnail", "title", "price", "rating", "reviews", "seller", score_col, "link"]]
    st.dataframe(
        show,
        hide_index=True,
        column_config={
            "thumbnail": st.column_config.ImageColumn("Image"),
            "title": "Product",
            "price": st.column_config.NumberColumn("Price (₹)", format="%.0f"),
            "reviews": st.column_config.NumberColumn("Reviews", format="%d"),
            "link": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )

with tab2:
    comp = price_comparison(filtered, min_offers=2)
    if comp.empty:
        st.info("No product with 2+ sellers was found for this query.")
    else:
        st.caption("Prices of the same product across different sellers (grouped by title similarity).")
        st.dataframe(
            comp,
            hide_index=True,
            column_config={
                "lowest_price": st.column_config.NumberColumn("Lowest (₹)", format="%.0f"),
                "highest_price": st.column_config.NumberColumn("Highest (₹)", format="%.0f"),
                "you_save": st.column_config.NumberColumn("You save (₹)", format="%.0f"),
                "link": st.column_config.LinkColumn("Link", display_text="Open"),
            },
        )

with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top 10 ranked products: price**")
        top = filtered.head(10)
        labels = [f"{i + 1}. {t[:30]}" for i, t in enumerate(top["title"])]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top["price"].values, y=labels, ax=ax, color="#4C78A8")
        ax.set_xlabel("Price (₹)")
        st.pyplot(fig)

    with c2:
        st.markdown("**Price vs Rating** (bubble size = reviews)")
        rated = filtered[filtered["rating"] > 0]
        if rated.empty:
            st.info("No rated products available.")
        else:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=rated, x="rating", y="price", size="reviews", sizes=(30, 400),
                            legend=False, ax=ax2)
            ax2.set_xlabel("Rating")
            ax2.set_ylabel("Price (₹)")
            st.pyplot(fig2)