# Design (không có UI: định dạng đầu ra)

## API (JSON)
| Endpoint | Trả về |
|---|---|
| `GET /customers/{id}` | phân khúc, xác suất churn, LTV 90 ngày, đặc trưng |
| `GET /recommend/{id}?k=10` | top k món từ ranker; khách mới nhận danh sách phổ biến |
| `GET /similar/{stock_code}` | món tương tự: EASE cho món phổ biến, TF-IDF cho món hiếm (<5 khách) |
| `POST /score` | churn, LTV, phân khúc cho khách mới khi đặc trưng tính ở upstream |

## Báo cáo
`artifacts/report.json` là nguồn số duy nhất; README trình bày bằng bảng Markdown tiếng Việt, số thập phân dùng dấu phẩy (0,367), tiền dùng £.
Mỗi bảng kết quả luôn có baseline đơn giản đứng cạnh mô hình, kèm một đoạn "nhận xét trung thực" khi mô hình không thắng rõ.
