import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the data."""
    df.dropna(inplace=True)
    return df
