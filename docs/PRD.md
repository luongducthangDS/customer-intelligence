# PRD: Customer Intelligence

## Bài toán
Một nhà bán sỉ quà tặng ở Anh (UCI Online Retail II, 12/2009–12/2011) cần: gợi ý sản phẩm cho từng khách, phân khúc khách hàng, dự báo churn và doanh thu (LTV) 90 ngày tới, đều đánh giá theo thời gian (không rò rỉ tương lai).

## Người dùng
- Đội CRM/marketing: danh sách gợi ý, phân khúc kèm hành động, xác suất churn.
- Người đọc portfolio: dự án phủ phần Recommendation/Segmentation trong JD. (unverified: suy ra từ AGENTS.md)

## Mục tiêu
1. Gợi ý theo candidate → LambdaRank, thắng baseline buy-again (đạt NDCG@10 0,367 so với 0,336).
2. Phân khúc RFM được kiểm chứng bằng hành vi thật 90 ngày sau.
3. Churn và LTV 90 ngày có so sánh trung thực với baseline đơn giản.
4. API phục vụ, Docker, CI có quality gate.

## Không làm
- Tune siêu tham số, MLflow (đã minh hoạ ở demand-forecasting).
- Monitoring/PSI cho phần này (chưa có).
- Dữ liệu ngoài RFM (kênh, web/app event).
