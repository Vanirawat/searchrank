# SearchRank

AI-Powered E-Commerce Product Search, Ranking & Price Comparison System

## Team
- Vani Rawat (Team Lead)
- Palak Kamboj
- Anugya Rawat

## Setup
1. Clone the repo
2. `python -m venv venv` and activate it
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your SerpApi key
5. `streamlit run app.py`

## Project Structure
- `src/` : core modules (fetch, clean, similarity, ranking, db)
- `data/` : raw and processed data (not tracked by git)
- `notebooks/` : EDA and experiments
- `docs/` : proposal and report
- `app.py` : Streamlit UI