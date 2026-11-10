"""Load and clean UCI Online Retail II (UK gift wholesaler, 2009-12 → 2011-12)."""
from pathlib import Path
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.csv"
CLEAN = ROOT / "data" / "clean.parquet"
URL = "https://huggingface.co/datasets/attik/Online-Retail-II-UCI/resolve/main/online_retail_II.csv"


def clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns={"Invoice": "invoice", "StockCode": "item", "Description": "desc",
                          "Quantity": "qty", "InvoiceDate": "date", "Price": "price",
                          "Customer ID": "customer", "Country": "country"})
    d = d.dropna(subset=["customer"]).drop_duplicates()
    d["item"] = d["item"].astype(str).str.upper().str.strip()
    # ponytail: real products start with a digit; drops POST, DOT, M, BANK CHARGES, gift vouchers, a few DCGS* items
    d = d[d["item"].str.match(r"^\d") & (d["price"] > 0)].copy()
    d["customer"] = d["customer"].astype("int64")
    d["invoice"] = d["invoice"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d["is_return"] = d["invoice"].str.startswith("C") | (d["qty"] < 0)
    d["revenue"] = d["qty"] * d["price"]
    return d.reset_index(drop=True)


def load() -> pd.DataFrame:
    if CLEAN.exists():
        return pd.read_parquet(CLEAN)
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    d = clean(pd.read_csv(RAW, encoding="latin-1", dtype={"Invoice": str, "StockCode": str}))
    d.to_parquet(CLEAN, index=False)
    return d


if __name__ == "__main__":
    d = load()
    print(d.shape, d["date"].min(), d["date"].max(), d["customer"].nunique(), d["item"].nunique())
