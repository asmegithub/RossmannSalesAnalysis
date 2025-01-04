import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates additional features from the data."""
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    return df
