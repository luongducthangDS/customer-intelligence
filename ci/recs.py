"""Product recommendation: candidates (EASE + buy-again + popular) → LightGBM LambdaRank."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightgbm import LGBMRanker

RANK_FEATS = ["ease", "ease_rank", "ui_cnt", "ui_days", "pop90", "pop_all", "repeat_rate", "price",
              "user_invoices", "user_items"]


class History:
    """Purchases strictly before `cutoff`, as a binary user×item matrix plus item / user stats."""

    def __init__(self, tx: pd.DataFrame, cutoff, min_users: int = 5):
        self.cutoff = cutoff = pd.Timestamp(cutoff)
        p = tx[(tx["date"] < cutoff) & ~tx["is_return"]]
        ui = p.groupby(["customer", "item"], as_index=False).agg(cnt=("invoice", "nunique"), last=("date", "max"))
        n_users = ui.groupby("item").size()
        self.items = pd.Index(sorted(n_users[n_users >= min_users].index))
        ui = ui[ui["item"].isin(self.items)]
        self.users = pd.Index(sorted(ui["customer"].unique()))
        ui = ui.assign(u=self.users.get_indexer(ui["customer"]), i=self.items.get_indexer(ui["item"]))
        self.ui = ui
        self.X = sp.csr_matrix((np.ones(len(ui), np.float32), (ui["u"], ui["i"])),
                               shape=(len(self.users), len(self.items)))
        recent = p[p["date"] >= cutoff - pd.Timedelta(days=90)]
        self.pop90 = recent.groupby("item")["customer"].nunique().reindex(self.items, fill_value=0).to_numpy(float)
        self.pop_all = n_users.reindex(self.items).to_numpy(float)
        self.repeat_rate = ui[ui["cnt"] >= 2].groupby("i").size().reindex(range(len(self.items)), fill_value=0).to_numpy() / self.pop_all
        self.price = p.groupby("item")["price"].median().reindex(self.items).to_numpy(float)
        self.user_invoices = p.groupby("customer")["invoice"].nunique().reindex(self.users).to_numpy(float)
        self.user_items = ui.groupby("u").size().to_numpy(float)

    def truth(self, tx, horizon=pd.Timedelta(days=90)) -> dict:
        """Items each known user buys in [cutoff, cutoff + horizon), restricted to known items."""
        w = tx[(tx["date"] >= self.cutoff) & (tx["date"] < self.cutoff + horizon) & ~tx["is_return"]]
        w = w[w["customer"].isin(self.users) & w["item"].isin(self.items)]
        u, i = self.users.get_indexer(w["customer"]), self.items.get_indexer(w["item"])
        return pd.Series(i).groupby(u).agg(set).to_dict()

    def bought(self) -> dict:
        return self.ui.groupby("u")["i"].agg(set).to_dict()


def ease(X: sp.csr_matrix, lam: float) -> np.ndarray:
    """EASE (Steck 2019): closed-form item-item weights with zero diagonal."""
    G = (X.T @ X).toarray().astype(np.float64)
    d = np.diag_indices_from(G)
    G[d] += lam
    P = np.linalg.inv(G)
    B = P / -np.diag(P)
    B[d] = 0
    return B.astype(np.float32)


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(scores, idx, 1), 1), 1)


def baseline_recs(h: History, B, users, k=10) -> dict:
    """Return {name: {u: [items]}} for popularity, buy-again, EASE."""
    pop = list(np.argsort(-h.pop90)[:k])
    ui = h.ui.sort_values(["u", "cnt", "last"], ascending=[True, False, False]).groupby("u")["i"].agg(list)
    again = {u: (ui.get(u, []) + [i for i in pop if i not in set(ui.get(u, []))])[:k] for u in users}
    S = h.X[users] @ B
    return {"popular": {u: pop for u in users}, "buy_again": again,
            "ease": dict(zip(users, topk(S, k).tolist()))}


def candidate_frame(h: History, B, users, n=50) -> pd.DataFrame:
    """Union of EASE top-n, the user's own past items and the n most popular items, with ranking features."""
    users = np.asarray(users)
    S = np.asarray(h.X[users] @ B)
    top = topk(S, n)
    pop = np.argsort(-h.pop90)[:n]
    own = h.ui.groupby("u")["i"].agg(np.array)
    rows_u, rows_i = [], []
    for k, u in enumerate(users):
        c = np.unique(np.concatenate([top[k], own.get(u, np.array([], int)), pop]))
        rows_u.append(np.full(len(c), k))
        rows_i.append(c)
        # ponytail: python loop over users (~5k, fine); vectorize if users reach 1e6
    k_idx, i_idx = np.concatenate(rows_u), np.concatenate(rows_i)
    f = pd.DataFrame({"u": users[k_idx], "i": i_idx, "ease": S[k_idx, i_idx]})
    f["ease_rank"] = f.groupby("u")["ease"].rank(ascending=False)
    hist = h.ui.set_index(["u", "i"])
    m = hist.reindex(pd.MultiIndex.from_arrays([f["u"], f["i"]]))
    f["ui_cnt"] = m["cnt"].fillna(0).to_numpy()
    f["ui_days"] = (h.cutoff - m["last"]).dt.days.to_numpy()
    for name in ("pop90", "pop_all", "repeat_rate", "price"):
        f[name] = getattr(h, name)[f["i"]]
    f["user_invoices"] = h.user_invoices[f["u"]]
    f["user_items"] = h.user_items[f["u"]]
    return f


def fit_ranker(h: History, B, truth: dict, seed=0) -> LGBMRanker:
    users = np.array(sorted(truth))
    f = candidate_frame(h, B, users)
    f["y"] = [int(i in truth[u]) for u, i in zip(f["u"], f["i"])]
    f = f[f.groupby("u")["y"].transform("max") > 0]  # groups without positives give no gradient
    m = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   random_state=seed, n_jobs=1, verbose=-1)  # ponytail: 1 thread, data is small; raise for >1e6 rows
    return m.fit(f[RANK_FEATS], f["y"], group=f.groupby("u", sort=False).size().to_numpy())


def ranker_recs(h: History, B, ranker, users, k=10) -> dict:
    f = candidate_frame(h, B, users)
    f["score"] = ranker.predict(f[RANK_FEATS])
    f = f.sort_values(["u", "score"], ascending=[True, False])
    return f.groupby("u")["i"].agg(lambda s: list(s[:k])).to_dict()


def evaluate(recs: dict, truth: dict, bought: dict, k=10) -> dict:
    """Recall@k, NDCG@k, HitRate@k over users in `truth`; recall split into repeat vs new items."""
    disc = 1 / np.log2(np.arange(2, k + 2))
    r, n, hit, rep, new = [], [], [], [], []
    for u, t in truth.items():
        rec = recs[u][:k]
        g = np.array([i in t for i in rec], float)
        r.append(g.sum() / len(t))
        n.append((g * disc[:len(g)]).sum() / disc[:min(k, len(t))].sum())
        hit.append(float(g.any()))
        old = bought.get(u, set())
        t_rep, t_new = t & old, t - old
        if t_rep:
            rep.append(len(t_rep & set(rec)) / len(t_rep))
        if t_new:
            new.append(len(t_new & set(rec)) / len(t_new))
    mean = lambda x: round(float(np.mean(x)), 4) if x else None
    return {"recall": mean(r), "ndcg": mean(n), "hit_rate": mean(hit),
            "recall_repeat": mean(rep), "recall_new": mean(new), "users": len(truth)}


if __name__ == "__main__":
    # self-check: EASE recovers a co-purchase pair
    X = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32))
    B = ease(X, 0.5)
    assert B[0, 1] > B[0, 2] and B[2, 2] == 0
    assert evaluate({0: [1, 2]}, {0: {1}}, {}, k=2)["ndcg"] == 1.0
    print("ok")
