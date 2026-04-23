import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger("agentic_rag.query_data")

# ── Global cache ──
_df_cache = None

def _get_df():
    """Lazy-load and cache the financial CSV."""
    global _df_cache
    if _df_cache is None:
        csv_path = Path("data/financials.csv")
        if not csv_path.exists():
            return None
        logger.info("Loading financials.csv (first call)...")
        _df_cache = pd.read_csv(csv_path)
    return _df_cache.copy()

def query_data(question):
    """Query structured financial data (revenue, margin, profit, EPS, headcount)."""
    try:
        df = _get_df()
        if df is None:
            return "Error: financials.csv not found."
        q = question.lower()
        
        companies = [c for c in ["infosys", "tcs", "wipro"] if c in q]
        years = [y for y in ["fy21", "fy22", "fy23", "fy24", "2021", "2022", "2023", "2024"] if y in q]
        # Normalize year entries
        years = ["fy" + y[2:] if y.startswith("20") else y for y in years]
        
        # Filter dataframe based on detected entities
        filtered_df = df
        if companies:
            filtered_df = filtered_df[filtered_df['company'].str.lower().isin(companies)]
        if years:
            filtered_df = filtered_df[filtered_df['year'].str.lower().isin(years)]
            
        res = ""
        metric = ""
        ascending = False
        
        if "highest" in q or "best" in q:
            ascending = False
        elif "lowest" in q or "worst" in q:
            ascending = True
            
        is_sort_query = "highest" in q or "best" in q or "lowest" in q or "worst" in q

        if "operating margin" in q:
            metric = "operating_margin_pct"
        elif "revenue" in q:
            metric = "revenue_crore"
        elif "net profit" in q:
            metric = "net_profit_crore"
        elif "eps" in q:
            metric = "eps_inr"
        elif "headcount" in q:
            metric = "headcount"
        elif "compare" in q or "all" in q:
            res = filtered_df.to_string(index=False)
        
        if metric:
            if is_sort_query:
                filtered_df = filtered_df.sort_values(by=metric, ascending=ascending)
                res = filtered_df.head(1).to_string(index=False)
            elif len(filtered_df) == 1:
                val = filtered_df.iloc[0][metric]
                res = str(val)
            else:
                res = filtered_df[['company', 'year', metric]].to_string(index=False)
                
        if res:
             return f"{res}\nSource: financials.csv"
             
        return f"No structured data found for: {question}.\nTry rephrasing or use search_docs for qualitative content."
        
    except Exception as e:
        return f"Error during data query: {str(e)}"
