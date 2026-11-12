"""Serving API. Reads artifacts written by `python -m ci.train` (dir overridable via CI_ARTIFACTS).

Env: CI_API_KEY (set -> every endpoint except /health needs header X-API-Key).
Mỗi request ghi một dòng log JSON (request_id, path, status, latency_ms, model_version) ra stdout.
"""
import hashlib
import hmac
import json
import logging
import math
import os
import statistics
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ci import customer as cu

ART = Path(os.environ.get("CI_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
API_KEY = os.environ.get("CI_API_KEY")
K_MAX = 20  # recs.parquet chỉ lưu top 20 mỗi khách
app = FastAPI(title="Customer Intelligence")
S: dict = {}
log = logging.getLogger("ci.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")
# ponytail: metrics trong bộ nhớ một tiến trình; nhiều worker thì đẩy sang Prometheus
LAT: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
COUNT: dict[str, int] = defaultdict(int)


class ArtifactsMissing(Exception):
    pass


def state():
    if not S:
        try:
            S["customers"] = pd.read_parquet(ART / "customers.parquet").set_index("customer")
            S["recs"] = pd.read_parquet(ART / "recs.parquet").set_index("customer")["items"]
            sim = pd.read_parquet(ART / "similar.parquet").set_index("item")
            S["similar"], S["desc"] = sim, sim["desc"]
            S["popular"] = json.loads((ART / "popular.json").read_text())
            S["models"] = joblib.load(ART / "models.joblib")
            S["version"] = hashlib.sha256((ART / "models.joblib").read_bytes()).hexdigest()[:12]
        except (FileNotFoundError, OSError) as e:
            S.clear()
            raise ArtifactsMissing(str(e)) from e
    return S


@app.exception_handler(ArtifactsMissing)
def _missing(_: Request, e: ArtifactsMissing):
    return JSONResponse({"detail": f"model artifacts not loaded, run `python -m ci.train`: {e}"}, status_code=503)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    route = request.scope.get("route")
    key = f"{request.method} {route.path if route else request.url.path}"
    COUNT[f"{key} {resp.status_code}"] += 1
    LAT[key].append(ms)
    resp.headers["x-request-id"] = rid
    log.info(json.dumps({"request_id": rid, "route": key, "status": resp.status_code, "latency_ms": ms,
                         "model_version": S.get("version")}))
    return resp


def auth(x_api_key: str | None = Header(None)):
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "invalid or missing X-API-Key")


def items(codes, k):
    d = state()["desc"]
    return [{"item": c, "desc": d.get(c, "")} for c in list(codes)[:k]]


@app.get("/health")
def health():
    s = state()
    return {"customers": len(s["customers"]), "items": len(s["similar"]), "model_version": s["version"]}


@app.get("/metrics", dependencies=[Depends(auth)])
def metrics():
    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, math.ceil(q * len(xs)) - 1)]
    return {"model_version": S.get("version"), "requests": dict(COUNT),
            "latency_ms": {k: {"n": len(v), "p50": statistics.median(v), "p95": pct(v, 0.95)}
                           for k, v in LAT.items() if v}}


@app.get("/customers/{customer_id}", dependencies=[Depends(auth)])
def customer(customer_id: int):
    c = state()["customers"]
    if customer_id not in c.index:
        raise HTTPException(404, "unknown customer")
    row = c.loc[customer_id]
    return {"customer": customer_id, "segment": int(row["segment"]), "churn_prob": round(float(row["churn_prob"]), 4),
            "ltv_90d": round(float(row["ltv_90d"]), 2), "model_version": state()["version"],
            "features": {k: None if pd.isna(v) else round(float(v), 3) for k, v in row[cu.FEATS].items()}}


@app.get("/recommend/{customer_id}", dependencies=[Depends(auth)])
def recommend(customer_id: int, k: int = Query(10, ge=1, le=K_MAX)):
    r = state()["recs"]
    if customer_id in r.index:
        return {"customer": customer_id, "source": "ranker", "items": items(r[customer_id], k)}
    return {"customer": customer_id, "source": "popular_cold_start", "items": items(state()["popular"], k)}


@app.get("/similar/{item}", dependencies=[Depends(auth)])
def similar(item: str, k: int = Query(10, ge=1, le=K_MAX)):
    s = state()["similar"]
    item = item.upper()[:32]
    if item not in s.index:
        raise HTTPException(404, "unknown item")
    return {"item": item, "source": s.loc[item, "source"], "similar": items(s.loc[item, "similar"], k)}


class Features(BaseModel):
    features: dict[str, float | None] = Field(max_length=64)


@app.post("/score", dependencies=[Depends(auth)])
def score(body: Features):
    """Score a customer that is not in the snapshot (e.g. features computed upstream)."""
    m = state()["models"]
    missing = [f for f in m["feats"] if f not in body.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    if any(body.features[f] is None for f in cu.RFM):
        raise HTTPException(422, f"{cu.RFM} must not be null")
    if any(v is not None and not math.isfinite(v) for v in body.features.values()):
        raise HTTPException(422, "features must be finite")
    X = pd.DataFrame([body.features])[m["feats"]].astype(float)
    return {"churn_prob": round(float(m["churn"].predict_proba(X)[0, 1]), 4),
            "ltv_90d": round(float(m["ltv"].predict(X)[0]), 2),
            "segment": int(cu.assign_segments(m["segments"], X).iat[0]), "model_version": state()["version"]}
