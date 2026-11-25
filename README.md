# CS232 – HT2: Hệ thống tìm ảnh tương tự

**Đề tài:** _(điền: nhu cầu thực tế)_
**Tiêu chí tương tự:** _(điền)_
**Thành viên:** _(Họ tên – MSSV)_

## Nội dung

| File | Vai trò |
|---|---|
| `core.py` | 2 phương pháp (histogram màu HSV, embedding CLIP), tìm top-5, đánh giá Hit@5 |
| `app.py` + `index.html` | Web app demo |
| `images/` | 50 ảnh tập tìm kiếm, tên file = ID (01.jpg … 50.jpg) |
| `sources.csv` | Nguồn ảnh |
| `ground_truth.csv` | 10 truy vấn + ID ảnh phù hợp (ghi trước khi chạy) |
| `results.md` | Kết quả 2 phương pháp + Hit@5 (do `core.py` tạo) |

## Cách chạy

```bash
pip install torch transformers pillow numpy
python core.py    # đánh giá -> results.md
python app.py     # demo tại http://localhost:8000
```

## Nhận xét – trường hợp xử lý chưa tốt

_(điền)_
