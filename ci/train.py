"""Backtest (train ≤ T1, test at T2 = last 90 days), then refit everything for serving.

python -m ci.train          # full data, writes artifacts/ + artifacts/report.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from ci import customer as cu, data, recs as rc

ART = data.ROOT / "artifacts"
LAMBDAS = (50, 200, 500, 1000, 2000)
D90 = pd.Timedelta(days=90)


def cutoffs(tx):
    end = tx["date"].max().normalize() + pd.Timedelta(days=1)
    t2, t1 = end - D90, end - 2 * D90
    train = [t1 - k * D90 for k in range(10) if t1 - k * D90 >= tx["date"].min() + pd.Timedelta(days=180)]
    return end, t2, t1, sorted(train)


def churn_model(seed=0):
    return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)  # ponytail: 1 thread, data is small; raise for >1e6 rows


def ltv_model(seed=0):
    return LGBMRegressor(objective="tweedie", n_estimators=400, learning_rate=0.03, num_leaves=15,
                         min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=seed, n_jobs=1, verbose=-1)  # ponytail: 1 thread, data is small; raise for >1e6 rows


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                         StandardScaler(), LogisticRegression(max_iter=2000))


def churn_metrics(y, p):
    top = p >= np.quantile(p, 0.9)
    return {"auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "lift_top10": round(y[top].mean() / y.mean(), 3)}


def ltv_metrics(y, p):
    top = p >= np.quantile(p, 0.9)
    return {"mae": round(mean_absolute_error(y, p), 1), "spearman": round(spearmanr(y, p)[0], 4),
            "top10_revenue_share": round(y[top].sum() / y.sum(), 4)}


def similar_items(tx, h: rc.History, B, k=10) -> pd.DataFrame:
    """EASE neighbours for known items; TF-IDF on description for items too rare for the matrix."""
    desc = tx.groupby("item")["desc"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
    known = np.array([i in h.items for i in desc.index])
    out = {}
    for row, item in enumerate(h.items):
        s = B[row].copy()
        s[row] = -np.inf
        out[item] = list(h.items[np.argsort(-s)[:k]])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(desc.fillna(""))
    if (~known).any():
        sim = (vec.transform(desc[~known]) @ vec.transform(desc[known]).T).toarray()
        for item, s in zip(desc.index[~known], sim):
            out[item] = list(desc.index[known][np.argsort(-s)[:k]])
    return pd.DataFrame({"item": list(out), "similar": list(out.values()), "source": [
        "ease" if i in h.items else "tfidf" for i in out]}).merge(desc.rename("desc"), left_on="item", right_index=True)


def main(tx=None, out: Path = ART, seed=0):
    t0 = time.time()
    tx = data.load() if tx is None else tx
    out.mkdir(parents=True, exist_ok=True)
    end, t2, t1, train_cuts = cutoffs(tx)
    report = {"cutoffs": {"train": [str(c.date()) for c in train_cuts], "test": str(t2.date()), "end": str(end.date())}}

    # ---- churn + LTV backtest
    tr, te = cu.dataset(tx, train_cuts), cu.dataset(tx, [t2])
    X, Xt = tr[cu.FEATS], te[cu.FEATS]
    yt = te["churn"].to_numpy()
    ch = churn_model(seed).fit(X, tr["churn"])
    lr = logreg().fit(X, tr["churn"])
    report["churn"] = {"train_rows": len(tr), "test_rows": len(te), "base_rate": round(yt.mean(), 4),
                       "lgbm": churn_metrics(yt, ch.predict_proba(Xt)[:, 1]),
                       "logreg": churn_metrics(yt, lr.predict_proba(Xt)[:, 1]),
                       "recency_rule": churn_metrics(yt, Xt["recency"].to_numpy(float))}
    lv = ltv_model(seed).fit(X, tr["ltv"])
    lt = te["ltv"].to_numpy()
    report["ltv"] = {"lgbm_tweedie": ltv_metrics(lt, lv.predict(Xt)),
                     "avg_90d_spend": ltv_metrics(lt, (Xt["mon_365"] / 4).to_numpy()),
                     "oracle_top10_share": ltv_metrics(lt, lt)["top10_revenue_share"]}

    # ---- segmentation at T2, validated against what happened in the next 90 days
    seg = cu.fit_segments(te[cu.FEATS], seed=seed)
    te["segment"] = cu.assign_segments(seg, te)
    prof = te.groupby("segment").agg(n=("churn", "size"), recency=("recency", "median"),
                                     freq_365=("freq_365", "median"), mon_365=("mon_365", "median"),
                                     churn_next90=("churn", "mean"), ltv_next90=("ltv", "mean")).round(2)
    report["segments"] = {"k": seg["k"], "silhouette": round(seg["silhouette"], 4),
                          "profile": prof.reset_index().to_dict("records")}

    # ---- recommendations: tune λ and train ranker on the T1 window, test on the T2 window
    hv = rc.History(tx, t1)
    truth_v, bought_v = hv.truth(tx), hv.bought()
    tune = {lam: rc.evaluate(rc.baseline_recs(hv, rc.ease(hv.X, lam), list(truth_v))["ease"], truth_v, bought_v)["ndcg"]
            for lam in LAMBDAS}
    lam = max(tune, key=tune.get)
    ranker = rc.fit_ranker(hv, rc.ease(hv.X, lam), truth_v, seed)
    ht = rc.History(tx, t2)
    Bt, truth_t, bought_t = rc.ease(ht.X, lam), ht.truth(tx), ht.bought()
    users = list(truth_t)
    recs = rc.baseline_recs(ht, Bt, users)
    recs["ranker"] = rc.ranker_recs(ht, Bt, ranker, users)
    cand = rc.candidate_frame(ht, Bt, users).groupby("u")["i"].agg(set)
    report["recs"] = {"ease_lambda_ndcg_val": tune, "lambda": lam, "items": len(ht.items),
                      "candidate_recall": round(float(np.mean([len(t & cand[u]) / len(t) for u, t in truth_t.items()])), 4),
                      **{k: rc.evaluate(v, truth_t, bought_t) for k, v in recs.items()}}

    # ---- refit on everything up to `end` and write serving artifacts
    full = cu.dataset(tx, train_cuts + [t2])
    ch_s = churn_model(seed).fit(full[cu.FEATS], full["churn"])
    lv_s = ltv_model(seed).fit(full[cu.FEATS], full["ltv"])
    snap = cu.features(tx, end)
    seg_s = cu.fit_segments(snap, seed=seed)
    snap["segment"] = cu.assign_segments(seg_s, snap)
    snap["churn_prob"] = ch_s.predict_proba(snap[cu.FEATS])[:, 1]
    snap["ltv_90d"] = lv_s.predict(snap[cu.FEATS])
    snap.reset_index().rename(columns={"index": "customer"}).to_parquet(out / "customers.parquet", index=False)
    joblib.dump({"churn": ch_s, "ltv": lv_s, "segments": seg_s, "feats": cu.FEATS}, out / "models.joblib")

    hs = rc.History(tx, end)
    Bs = rc.ease(hs.X, lam)
    rank_s = rc.fit_ranker(ht, Bt, truth_t, seed)
    all_users = np.arange(len(hs.users))
    top = rc.ranker_recs(hs, Bs, rank_s, all_users, k=20)
    pd.DataFrame({"customer": hs.users[all_users], "items": [list(hs.items[top[u]]) for u in all_users]}) \
        .to_parquet(out / "recs.parquet", index=False)
    similar_items(tx, hs, Bs).to_parquet(out / "similar.parquet", index=False)
    (out / "popular.json").write_text(json.dumps(list(hs.items[np.argsort(-hs.pop90)[:20]])))

    report["seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
