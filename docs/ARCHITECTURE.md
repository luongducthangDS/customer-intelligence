# Architecture

## Stack
Python 3.11 (uv) · pandas/pyarrow · scikit-learn (KMeans, logistic regression, TF-IDF) · LightGBM (classifier, tweedie, LambdaRank) · EASE (numpy) · FastAPI · pytest · Docker · GitHub Actions.

## Thư mục
```
ci/data.py       tải mirror HF attik/Online-Retail-II-UCI, làm sạch → data/clean.parquet (gitignored)
ci/customer.py   đặc trưng RFM tại mốc cắt (chỉ dùng dòng date < c), nhãn churn/LTV 90 ngày, KMeans RFM
ci/recs.py       EASE, buy-again, phổ biến, candidate, LambdaRank, metric (Recall/NDCG/HitRate@10)
ci/train.py      backtest theo mốc cắt, chọn λ EASE, train ranker, fit lại để serving → artifacts/
ci/api.py        FastAPI: /customers/{id}, /recommend/{id}, /similar/{stock_code}, /score, /health, /metrics; API key, log JSON
ci/gate.py       regression gate: report.json so với eval/baseline.json + model phải thắng luật đơn giản
tests/test_pipeline.py  shop tổng hợp 300 khách: không rò rỉ, nhãn đúng, ranker thắng phổ biến, API chạy
```

## Luồng dữ liệu
```
data/clean.parquet → 5 mốc cắt train (cách 90 ngày) → validation 2011-06-13→09-11 → test 2011-09-11→12-10
  customer.py: đặc trưng + nhãn → churn (LR, LightGBM), LTV (tweedie), KMeans
  recs.py: candidate = top50 EASE ∪ món đã mua ∪ top50 phổ biến → LambdaRank
train.py (serving): fit lại đến T2, chấm snapshot cuối, gợi ý tính trước top 20/khách → artifacts/ → api.py tra cứu
```
