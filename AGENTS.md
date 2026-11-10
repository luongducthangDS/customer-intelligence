# customer-intelligence: agent notes

Đây là dự án phủ phần Recommendation và Segmentation trong JD (gợi ý, phân khúc, churn, LTV). Chưa git init. Kết quả và cách chạy xem `README.md`.

## 2026-11-09: dựng end to end

- **Dữ liệu:** UCI Online Retail II, mirror HF `attik/Online-Retail-II-UCI`. `ci/data.py` tải và làm sạch vào `data/clean.parquet`; `data/` nằm trong gitignore.
- **Pipeline:**
  - `ci/customer.py`: đặc trưng tại mốc cắt, nhãn churn/LTV 90 ngày, KMeans RFM
  - `ci/recs.py`: EASE, buy-again, candidate, LambdaRank, metric
  - `ci/train.py`: backtest, fit lại để serving, ghi `artifacts/`
  - `ci/api.py`: FastAPI
- **Kết quả test** (cắt ngày 2011-09-11):
  - churn: LightGBM AUC 0,764, logistic regression 0,769, luật recency 0,707
  - LTV: LightGBM ngang baseline
  - gợi ý: ranker NDCG@10 0,367, buy-again 0,336, EASE 0,265, phổ biến 0,104
- **Đã kiểm:** `pytest` 5 passed (khoảng 45 s); API chạy trên artifact thật bằng TestClient; `docker build` rồi `docker run` thành công, `/health` và `/recommend` trả kết quả đúng.

## Gotchas

- Dùng `.venv` riêng của thư mục này (uv, Python 3.11). Lệnh `python` mặc định của shell trỏ vào venv của DocuMindAI.
- Khi máy đang có job nặng khác chạy (ví dụ PhoBERT của `vietnamese-nlp`), LightGBM chạy nhiều luồng bị oversubscription và chậm gấp 3–4 lần. Vì vậy mô hình đang để `n_jobs=1`.
- `overdue = recency / (ipt_mean + 1)`. Phải giữ `+1` vì các hoá đơn mua cùng ngày cho `ipt_mean = 0`, chia ra inf và làm logistic regression lỗi.
- Bản serving fit lại phân khúc trên snapshot cuối và silhouette chọn k = 3, khác k = 4 ở bản backtest.

## Còn lại

- Candidate recall mới 45 %; nên thêm nguồn candidate theo đồng mua và theo danh mục.
- Chưa có monitoring hay MLflow cho dự án này.
