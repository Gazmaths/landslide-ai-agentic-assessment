import pandas as pd

def load_csv_or_none(path: str | None):
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None