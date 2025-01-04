import pandas as pd

def analyze_sales(df: pd.DataFrame):
    """Performs sales analysis."""
    top_sales = df.groupby('Store')['Sales'].sum().sort_values(ascending=False).head(10)
    return top_sales
