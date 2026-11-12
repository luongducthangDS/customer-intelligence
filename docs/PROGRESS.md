# Progress

## Trạng thái (2026-11-10)
Xong end to end, public tại github.com/luongducthangDS/customer-intelligence, CI pass.

Kết quả test (cắt 2011-09-11, `artifacts/report.json`):
- Churn: logistic regression AUC 0,769, LightGBM 0,764, luật recency 0,707.
- LTV: LightGBM MAE £495,9 so với baseline £491,8, tức là ngang hoặc kém nhẹ baseline.
- Gợi ý: LambdaRank NDCG@10 0,367, buy-again 0,336, EASE 0,265, phổ biến 0,104.
- Phân khúc k = 4 (silhouette 0,36): churn 90 ngày sau từ 9 % (Champions) đến 72 % (Đã rời bỏ).

Đã kiểm: `pytest` 7 passed (~30 s); train lại 12/11 (268 s) ra đúng từng số như report cũ; `python -m ci.gate` PASS; `docker run` với `CI_API_KEY`: thiếu key 401, `k=500` 422, `/metrics` và log JSON có request id + model version.

## Điểm yếu đã biết
- Candidate recall 45 %; recall món mới rất thấp (0,006).
- LTV không thắng baseline; churn chỉ hơn luật recency 6 điểm AUC.
- Chưa tune, chưa MLflow; chưa monitoring drift (đã có log request + `/metrics` vận hành).

## Bước tiếp theo
Mục đầu tiên chưa tick trong `TASKS.md`: thêm nguồn candidate.
