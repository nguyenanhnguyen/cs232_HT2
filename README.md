# CS232 – HT2: Tìm ảnh phương tiện cùng loại trong điều kiện thiếu sáng

**Thành viên:** _(Họ tên – MSSV)_

## Bài toán
Người quản lý một kho ảnh chụp ban đêm có một ảnh phương tiện (vd. xe buýt) và muốn tìm các ảnh **cùng loại phương tiện** trong kho mà không phải xem từng tấm.

- **Tiêu chí tương tự:** cùng loại phương tiện (xe đạp, thuyền, xe buýt, ô tô, xe máy); không xét màu, góc chụp hay ánh sáng.
- **Câu hỏi so sánh:** trong ảnh thiếu sáng, màu chủ yếu đến từ ánh đèn chứ không phải từ đối tượng. Vậy histogram màu hay embedding CLIP tìm đúng loại phương tiện tốt hơn, và mỗi cách thất bại vì sao?

## Dữ liệu
- **Nguồn:** ExDark – Loh, Y. P., & Chan, C. S. (2019). *Getting to know low-light images with the Exclusively Dark dataset.* CVIU, 178, 30–42. Tải từ: _(dán link Kaggle đã tải)_
- **Tập tìm kiếm:** 50 ảnh = 5 lớp × 10 ảnh, trong `images/`, ID dạng `bus_03`.
- **Truy vấn:** 10 ảnh = 5 lớp × 2 ảnh, nằm **ngoài** tập tìm kiếm, trong `queries/` (`q01` … `q10`).
- **Đáp án:** 10 ảnh cùng lớp với truy vấn, ghi trong `ground_truth.csv` trước khi chạy so sánh.
- Tên file gốc của từng ảnh: `sources.csv`.

## Hai phương pháp
| | Histogram màu | CLIP embedding |
|---|---|---|
| Biểu diễn | Tỉ lệ pixel theo ô màu HSV 8×4×4 = 128 chiều | Vector nội dung ảnh 512 chiều (CLIP ViT-B/32) |
| So khớp | Histogram intersection | Cosine similarity |
| Nắm bắt | Phân bố màu, độ sáng | Ngữ nghĩa: trong ảnh có vật gì |

## Cách chạy
```bash
pip install torch transformers pillow numpy

# 1. Chọn tay từ dataset/archive/: chép 10 ảnh/lớp vào images/<Lớp>/, 2 ảnh/lớp khác vào queries/<Lớp>/
# 2. Đổi tên thành ID, tạo sources.csv + ground_truth.csv
python core.py prepare
# 3. Chạy 10 truy vấn bằng 2 phương pháp -> results.md
python core.py
# 4. Demo web tại http://localhost:8000
python app.py
```

| File | Vai trò |
|---|---|
| `core.py` | 2 phương pháp, tìm top-5, chuẩn bị dữ liệu, đánh giá Hit@5 |
| `app.py` | Web server (thư viện chuẩn của Python) |
| `index.html` | Giao diện: tải ảnh lên, xem top-5 của 2 phương pháp cạnh nhau |

## Kết quả
Bảng đầy đủ: [results.md](results.md). Hit@5 = số truy vấn có ≥ 1 ảnh đúng trong top 5, trên 10 truy vấn.

| | Histogram màu | CLIP |
|---|---|---|
| Hit@5 | _/10 | _/10 |

_Mốc tham chiếu: mỗi truy vấn có 10/50 ảnh đúng, nên chọn ngẫu nhiên 5 ảnh đã có ≈ 69% khả năng trúng ít nhất 1 ảnh. Vì vậy bảng ghi thêm số ảnh đúng trong top 5._

## Nhận xét – một trường hợp xử lý chưa tốt
_(điền sau khi chạy: truy vấn nào, phương pháp nào sai, sai thành lớp nào, vì sao)_
