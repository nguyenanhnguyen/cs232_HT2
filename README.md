# CS232 - HT2: Tìm ảnh phương tiện cùng loại trong điều kiện thiếu sáng

Thành viên: (Họ tên - MSSV)


## Bài toán

Cho một ảnh phương tiện chụp trong điều kiện thiếu sáng, hệ thống tìm 5 ảnh cùng loại
phương tiện trong một kho 50 ảnh.

Tiêu chí tương tự: cùng loại phương tiện (xe đạp, thuyền, xe buýt, ô tô, xe máy),
không xét màu sắc, góc chụp hay ánh sáng.

Câu hỏi so sánh: trong ảnh tối, màu chủ yếu đến từ ánh đèn chứ không phải từ đối tượng.
Histogram màu hay embedding CLIP tìm đúng loại phương tiện tốt hơn?


## Dữ liệu

Nguồn: ExDark - Loh, Y. P., & Chan, C. S. (2019). Getting to know low-light images with
the Exclusively Dark dataset. Computer Vision and Image Understanding, 178, 30-42.
Link tải: (điền link Kaggle)

    images/            50 ảnh tìm kiếm, 5 lớp x 10 ảnh, ID dạng bus_03
    queries/           10 ảnh truy vấn, 5 lớp x 2 ảnh, nằm ngoài tập tìm kiếm
    sources.csv        tên file gốc của từng ảnh
    ground_truth.csv   ảnh đúng của từng truy vấn (các ảnh cùng lớp), ghi trước khi chạy


## Phương pháp

Histogram màu: tỉ lệ pixel theo ô màu HSV 8x4x4 (128 chiều), so khớp bằng histogram intersection.

CLIP embedding: vector nội dung ảnh 512 chiều (CLIP ViT-B/32), so khớp bằng cosine similarity.

Nếu ảnh truy vấn trùng một ảnh trong tập tìm kiếm, ảnh đó bị loại khỏi kết quả.


## Cách chạy

    pip install torch transformers pillow numpy

    python core.py prepare    # đổi tên ảnh thành ID, tạo sources.csv và ground_truth.csv
    python core.py            # chạy 10 truy vấn với 2 phương pháp, ghi results.md
    python app.py             # demo web tại http://localhost:8000

Trước khi chạy prepare, chép ảnh đã chọn vào images/<Lớp>/ và queries/<Lớp>/.
Lệnh prepare chỉ chạy một lần.


## Kết quả

Hit@5 là số truy vấn có ít nhất một ảnh đúng trong 5 kết quả đầu, trên 10 truy vấn.
Bảng chi tiết: results.md

    Histogram màu:   _/10
    CLIP embedding:  _/10

Mỗi truy vấn có 10/50 ảnh đúng, nên chọn ngẫu nhiên 5 ảnh đã có khoảng 69% khả năng
trúng ít nhất một ảnh. Vì vậy bảng ghi thêm số ảnh đúng trong top 5.


## Nhận xét - trường hợp xử lý chưa tốt

(Truy vấn nào, phương pháp nào sai, sai thành lớp nào, vì sao)
