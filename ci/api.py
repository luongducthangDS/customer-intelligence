"""Serving API. Reads artifacts written by `python -m ci.train` (dir overridable via CI_ARTIFACTS)."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ci import customer as cu

ART = Path(os.environ.get("CI_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
app = FastAPI(title="Customer Intelligence")
S: dict = {}


def state():
    if not S:
        S["customers"] = pd.read_parquet(ART / "customers.parquet").set_index("customer")
        S["recs"] = pd.read_parquet(ART / "recs.parquet").set_index("customer")["items"]
        sim = pd.read_parquet(ART / "similar.parquet").set_index("item")
        S["similar"], S["desc"] = sim, sim["desc"]
        S["popular"] = json.loads((ART / "popular.json").read_text())
        S["models"] = joblib.load(ART / "models.joblib")
    return S


def items(codes, k):
    d = state()["desc"]
    return [{"item": c, "desc": d.get(c, "")} for c in list(codes)[:k]]


@app.get("/health")
def health():
    s = state()
    return {"customers": len(s["customers"]), "items": len(s["similar"])}


@app.get("/customers/{customer_id}")
def customer(customer_id: int):
    c = state()["customers"]
    if customer_id not in c.index:
        raise HTTPException(404, "unknown customer")
    row = c.loc[customer_id]
    return {"customer": customer_id, "segment": int(row["segment"]), "churn_prob": round(float(row["churn_prob"]), 4),
            "ltv_90d": round(float(row["ltv_90d"]), 2), "features": {k: None if pd.isna(v) else round(float(v), 3) for k, v in row[cu.FEATS].items()}}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, k: int = 10):
    r = state()["recs"]
    if customer_id in r.index:
        return {"customer": customer_id, "source": "ranker", "items": items(r[customer_id], k)}
    return {"customer": customer_id, "source": "popular_cold_start", "items": items(state()["popular"], k)}


@app.get("/similar/{item}")
def similar(item: str, k: int = 10):
    s = state()["similar"]
    item = item.upper()
    if item not in s.index:
        raise HTTPException(404, "unknown item")
    return {"item": item, "source": s.loc[item, "source"], "similar": items(s.loc[item, "similar"], k)}


class Features(BaseModel):
    features: dict[str, float | None]


@app.post("/score")
def score(body: Features):
    """Score a customer that is not in the snapshot (e.g. features computed upstream)."""
    m = state()["models"]
    missing = [f for f in m["feats"] if f not in body.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    if any(body.features[f] is None for f in cu.RFM):
        raise HTTPException(422, f"{cu.RFM} must not be null")
    X = pd.DataFrame([body.features])[m["feats"]].astype(float)
    return {"churn_prob": round(float(m["churn"].predict_proba(X)[0, 1]), 4),
            "ltv_90d": round(float(m["ltv"].predict(X)[0]), 2),
            "segment": int(cu.assign_segments(m["segments"], X).iat[0])}
