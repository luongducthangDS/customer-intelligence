"""Customer snapshot features, churn / LTV labels, RFM segmentation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HORIZON = pd.Timedelta(days=90)
FEATS = ["recency", "tenure", "frequency", "monetary", "avg_basket", "freq_90", "mon_90",
         "freq_365", "mon_365", "ipt_mean", "ipt_std", "overdue", "n_items", "return_ratio", "uk", "mon_ly"]
RFM = ["recency", "freq_365", "mon_365"]


def invoices(tx: pd.DataFrame) -> pd.DataFrame:
    p = tx[~tx["is_return"]]
    return p.groupby(["customer", "invoice"], as_index=False).agg(date=("date", "min"), value=("revenue", "sum"))


def features(tx: pd.DataFrame, cutoff) -> pd.DataFrame:
    """One row per customer active in the 365 days before `cutoff`. Uses only rows with date < cutoff."""
    cutoff = pd.Timestamp(cutoff)
    h = tx[tx["date"] < cutoff]
    inv = invoices(h)
    active = inv.loc[inv["date"] >= cutoff - pd.Timedelta(days=365), "customer"].unique()
    inv = inv[inv["customer"].isin(active)].sort_values("date")
    g = inv.groupby("customer")
    f = pd.DataFrame({
        "recency": (cutoff - g["date"].max()).dt.days,
        "tenure": (cutoff - g["date"].min()).dt.days,
        "frequency": g.size(),
        "monetary": g["value"].sum(),
        "avg_basket": g["value"].mean(),
    })
    for days in (90, 365):
        w = inv[inv["date"] >= cutoff - pd.Timedelta(days=days)].groupby("customer")
        f[f"freq_{days}"] = w.size()
        f[f"mon_{days}"] = w["value"].sum()
    gaps = g["date"].diff().dt.days
    f["ipt_mean"] = gaps.groupby(inv["customer"]).mean()
    f["ipt_std"] = gaps.groupby(inv["customer"]).std()
    f["overdue"] = f["recency"] / (f["ipt_mean"] + 1)  # +1: same-day invoices give ipt 0
    hc = h[h["customer"].isin(active)]
    f["n_items"] = hc[~hc["is_return"]].groupby("customer")["item"].nunique()
    f["return_ratio"] = hc[hc["is_return"]].groupby("customer")["revenue"].sum().abs() / f["monetary"]
    f["uk"] = ((hc["country"] == "United Kingdom").groupby(hc["customer"]).mean() > 0.5).astype(int)
    ly = inv[(inv["date"] >= cutoff - pd.Timedelta(days=365)) & (inv["date"] < cutoff - pd.Timedelta(days=275))]
    f["mon_ly"] = ly.groupby("customer")["value"].sum()  # same 90-day window last year: Q4 seasonality
    f[["freq_90", "mon_90", "return_ratio", "mon_ly"]] = f[["freq_90", "mon_90", "return_ratio", "mon_ly"]].fillna(0)
    return f[FEATS]  # ipt_* / overdue stay NaN for one-invoice customers; LightGBM handles NaN


def labels(tx: pd.DataFrame, cutoff, customers) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    w = tx[(tx["date"] >= cutoff) & (tx["date"] < cutoff + HORIZON) & ~tx["is_return"]]
    ltv = w.groupby("customer")["revenue"].sum().reindex(customers, fill_value=0.0)
    return pd.DataFrame({"churn": (ltv <= 0).astype(int), "ltv": ltv.clip(lower=0)}, index=customers)


def dataset(tx, cutoffs) -> pd.DataFrame:
    parts = []
    for c in cutoffs:
        f = features(tx, c)
        parts.append(f.join(labels(tx, c, f.index)).assign(cutoff=pd.Timestamp(c)))
    return pd.concat(parts)


def fit_segments(f: pd.DataFrame, ks=range(3, 8), seed=0):
    """KMeans on standardized log RFM; k by silhouette. Segment 0 = highest value."""
    scaler = StandardScaler().fit(np.log1p(f[RFM]))
    X = scaler.transform(np.log1p(f[RFM]))
    best = max((silhouette_score(X, KMeans(k, n_init=10, random_state=seed).fit_predict(X),
                                 sample_size=min(len(X), 3000), random_state=seed), k) for k in ks)
    km = KMeans(best[1], n_init=10, random_state=seed).fit(X)
    c = km.cluster_centers_
    order = np.argsort(-(-c[:, 0] + c[:, 1] + c[:, 2]))  # value = recent + frequent + high spend
    remap = {old: new for new, old in enumerate(order)}
    return {"scaler": scaler, "km": km, "remap": remap, "k": best[1], "silhouette": best[0]}


def assign_segments(seg, f: pd.DataFrame) -> pd.Series:
    raw = seg["km"].predict(seg["scaler"].transform(np.log1p(f[RFM])))
    return pd.Series([seg["remap"][r] for r in raw], index=f.index, name="segment")
