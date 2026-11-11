# Tasks

## Đã xong (2026-11-09 → 10)
- [x] Dữ liệu, đặc trưng tại mốc cắt, nhãn churn/LTV
- [x] EASE, buy-again, candidate, LambdaRank, phân khúc RFM
- [x] API, Docker, CI có quality gate, repo public

## Tiếp theo (lần lượt từ trên xuống)
- [ ] Thêm nguồn candidate đồng mua trong cùng hoá đơn và theo danh mục (candidate recall đang 45 %, là trần của ranker)
- [ ] Monitoring: PSI của phân phối điểm churn theo thời gian (cách làm như `fraud-graph/`)
- [ ] Profile `python -m ci.train` (~10 phút, chủ yếu ở candidate và ma trận EASE)
