"""Leakage, label and end-to-end pipeline checks on a small synthetic shop (runs in CI, no download)."""
import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ci import customer as cu, data, recs as rc, train


def synth(n_cust=300, seed=0):
    rng = np.random.default_rng(seed)
    start, rows = pd.Timestamp("2022-01-01"), []
    for c in range(n_cust):
        group = c % 3
        fav = rng.choice(range(group * 20, group * 20 + 20), 8, replace=False)
        for k, day in enumerate(np.sort(rng.integers(0, 730, rng.integers(1, 15)))):
            for it in rng.choice(fav, rng.integers(1, 5), replace=False):
                rows.append((f"{c}-{k}", f"{10000 + it}", f"ITEM {it} GROUP {group}", int(rng.integers(1, 10)),
                             start + pd.Timedelta(days=int(day)), 2.5, float(c + 1000),
                             "United Kingdom" if c % 4 else "France"))
    rows.append(("C1-0", "10001", "ITEM 1 GROUP 0", -2, start + pd.Timedelta(days=400), 2.5, 1001.0, "France"))
    rows.append(("9-9", "POST", "POSTAGE", 1, start + pd.Timedelta(days=10), 18.0, 1009.0, "France"))
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return data.clean(pd.DataFrame(rows, columns=cols))


@pytest.fixture(scope="module")
def tx():
    return synth()


def test_clean_drops_non_products_and_flags_returns(tx):
    assert "POST" not in set(tx["item"])
    assert tx.loc[tx["invoice"] == "C1-0", "is_return"].all()


def test_features_ignore_the_future(tx):
    cutoff = pd.Timestamp("2023-03-01")
    before = cu.features(tx, cutoff)
    future = tx["date"] >= cutoff
    perturbed = tx.copy()
    perturbed.loc[future, "revenue"] *= 100
    perturbed.loc[future, "item"] = "99999"
    pd.testing.assert_frame_equal(before, cu.features(perturbed, cutoff))


def test_labels(tx):
    cutoff = pd.Timestamp("2023-03-01")
    lab = cu.labels(tx, cutoff, cu.features(tx, cutoff).index)
    w = tx[(tx["date"] >= cutoff) & (tx["date"] < cutoff + cu.HORIZON) & ~tx["is_return"]]
    buyers = set(w["customer"])
    assert ((lab["churn"] == 0) == lab.index.isin(buyers)).all()
    c = next(iter(buyers & set(lab.index)))
    assert lab.loc[c, "ltv"] == pytest.approx(w.loc[w["customer"] == c, "revenue"].sum())


def test_history_truth_is_after_cutoff(tx):
    h = rc.History(tx, "2023-03-01")
    assert h.ui["last"].max() < h.cutoff
    assert h.X.shape == (len(h.users), len(h.items))


def test_end_to_end_train_and_api(tx, tmp_path, monkeypatch):
    report = train.main(tx, tmp_path)
    assert report["churn"]["lgbm"]["auc"] > 0.5
    assert report["recs"]["ranker"]["recall"] > report["recs"]["popular"]["recall"]  # groups make EASE/ranker useful

    monkeypatch.setenv("CI_ARTIFACTS", str(tmp_path))
    from ci import api
    api = importlib.reload(api)
    client = TestClient(api.app)
    cust = int(pd.read_parquet(tmp_path / "customers.parquet")["customer"].iat[0])
    assert client.get("/health").status_code == 200
    body = client.get(f"/customers/{cust}").json()
    assert 0 <= body["churn_prob"] <= 1
    rec = client.get(f"/recommend/{cust}?k=5").json()
    assert rec["source"] == "ranker" and len(rec["items"]) == 5
    assert client.get("/recommend/1?k=3").json()["source"] == "popular_cold_start"
    assert client.get("/similar/10001").status_code == 200
    assert client.post("/score", json={"features": body["features"]}).status_code == 200
    assert client.post("/score", json={"features": {"recency": 1}}).status_code == 422
    assert client.get(f"/recommend/{cust}?k=500").status_code == 422  # k vượt top 20 đã tính sẵn
    m = client.get("/metrics").json()
    assert m["model_version"] and m["latency_ms"]["GET /recommend/{customer_id}"]["n"] >= 2
    assert client.get("/health", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"

    monkeypatch.setattr(api, "API_KEY", "k3y")
    assert client.get("/health").status_code == 200  # health mở cho load balancer
    assert client.get(f"/customers/{cust}").status_code == 401
    assert client.get(f"/customers/{cust}", headers={"X-API-Key": "sai"}).status_code == 401
    assert client.get(f"/customers/{cust}", headers={"X-API-Key": "k3y"}).status_code == 200


def test_missing_artifacts_is_503(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_ARTIFACTS", str(tmp_path))
    from ci import api
    api = importlib.reload(api)
    assert TestClient(api.app).get("/health").status_code == 503


def test_gate():
    from ci import gate
    base = json.loads(gate.BASELINE.read_text())
    rep = json.loads((gate.ROOT / "artifacts" / "report.json").read_text())
    assert gate.check(rep, base) == []  # report đã commit phải qua gate
    worse = json.loads(json.dumps(rep))
    worse["recs"]["ranker"]["ndcg"] = rep["recs"]["buy_again"]["ndcg"] - 0.01
    worse["churn"]["lgbm"]["auc"] -= 0.05
    fails = gate.check(worse, base)
    assert any("recs.ranker.ndcg" in f and "baseline" in f for f in fails)
    assert any("không thắng recs.buy_again.ndcg" in f for f in fails)
    assert any("churn.lgbm.auc" in f for f in fails)
