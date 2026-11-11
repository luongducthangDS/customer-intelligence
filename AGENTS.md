# customer-intelligence: agent notes

Dự án phủ phần Recommendation và Segmentation trong JD (gợi ý, phân khúc, churn, LTV). Public tại github.com/luongducthangDS/customer-intelligence, CI pass.

Đọc trước khi làm: [docs/RULES.md](docs/RULES.md) > [docs/PRD.md](docs/PRD.md) > [ARCHITECTURE](docs/ARCHITECTURE.md), [DESIGN](docs/DESIGN.md), [TASKS](docs/TASKS.md), [PROGRESS](docs/PROGRESS.md).

## Trạng thái (2026-11-11)
- Xong end to end 09–10/11: ranker NDCG@10 0,367 so với buy-again 0,336; churn AUC 0,764–0,769; LTV ngang baseline. Chi tiết ở `docs/PROGRESS.md`.
- Việc tiếp theo: mục đầu tiên chưa tick trong `docs/TASKS.md`.
- `python -m ci.data` tải dữ liệu, `python -m ci.train` (~10 phút) ghi `artifacts/` trước khi chạy API.

## Gotchas
- Dùng `.venv` riêng (uv, Python 3.11); `python` mặc định của shell trỏ vào venv DocuMindAI.
- Job nặng khác đang chạy (PhoBERT…) làm LightGBM chậm 3–4 lần; vì vậy để `n_jobs=1`.
- Serving fit lại phân khúc trên snapshot cuối, silhouette chọn k = 3 (backtest là k = 4).
