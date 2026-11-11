# Rules

## Bất biến (không gỡ)
- Đặc trưng tại mốc `c` chỉ dùng dòng `date < c`; test `test_features_ignore_the_future` phải luôn pass.
- Quality gate trong CI: ranker phải thắng baseline phổ biến. Không nới gate để CI xanh.
- `overdue = recency / (ipt_mean + 1)`: giữ `+1` (hoá đơn cùng ngày cho `ipt_mean = 0`, chia ra inf làm hỏng logistic regression).
- LightGBM để `n_jobs=1` (chạy song song với job nặng khác bị oversubscription, chậm 3–4 lần).
- Không chỉnh cửa sổ validation/test để số đẹp hơn.

## Quy ước
- Dùng `.venv` riêng (uv, Python 3.11); `python` mặc định của shell là venv của DocuMindAI.
- Số trong README lấy từ `artifacts/report.json`.
- Bản serving fit lại có thể chọn k khác bản backtest (đang là k = 3 so với k = 4); khi ghi số phân khúc phải nói rõ là bản nào.

## AI không bao giờ được
- Commit `data/` hoặc artifact model lớn.
- Báo LTV là "tốt hơn baseline" khi MAE đang kém baseline.
- Push khi chưa được user cho phép.
