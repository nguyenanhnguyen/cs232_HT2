"""CS232 HT2 - Tìm ảnh phương tiện cùng loại trong điều kiện thiếu sáng (ExDark).

Hai cách biểu diễn ảnh:
  hist : histogram màu HSV 8x4x4 = 128 chiều, so khớp bằng histogram intersection
  clip : embedding CLIP ViT-B/32 = 512 chiều, so khớp bằng cosine similarity

Cách dùng:
  python core.py prepare   # đổi tên ảnh đã chọn -> ID, tạo sources.csv + ground_truth.csv
  python core.py           # chạy 10 truy vấn bằng 2 phương pháp -> results.md (Hit@5)
"""
import csv
import hashlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")  # in được tiếng Việt trên console Windows

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "images"    # 50 ảnh tập tìm kiếm
Q_DIR = ROOT / "queries"     # 10 ảnh truy vấn (nằm ngoài tập tìm kiếm)
EXTS = {".jpg", ".jpeg", ".png"}
CLIP_NAME = "openai/clip-vit-base-patch32"
METHODS = {"hist": "Histogram màu", "clip": "CLIP embedding"}
DATASET = "ExDark - Exclusively Dark Image Dataset (Loh & Chan, 2019)"


def load_image(data):
    """bytes -> ảnh RGB (xoay đúng chiều theo EXIF)."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


# ================= Phương pháp 1: histogram màu =================
def hist_feature(img, bins=(8, 4, 4)):
    """Đếm tỉ lệ pixel rơi vào từng ô màu HSV (8 mức Hue x 4 Saturation x 4 Value)."""
    img = img.copy()
    img.thumbnail((256, 256))  # thu nhỏ cho nhanh, không ảnh hưởng tỉ lệ màu
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    h, _ = np.histogramdd(hsv, bins=bins, range=[(0, 256)] * 3)
    return h.flatten() / h.sum()


def hist_scores(db, q):
    """Histogram intersection: tổng phần chung của 2 histogram, 1 = giống hệt, 0 = khác hẳn."""
    return np.minimum(db, q).sum(axis=1)


# ================= Phương pháp 2: embedding CLIP =================
_clip = None


def clip_feature(imgs):
    """Ảnh -> vector 512 chiều mô tả nội dung (đã chuẩn hoá độ dài = 1)."""
    global _clip
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if _clip is None:
        _clip = CLIPModel.from_pretrained(CLIP_NAME).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)
    model, proc = _clip
    with torch.no_grad():
        out = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
    emb = out if isinstance(out, torch.Tensor) else out.pooler_output  # tương thích nhiều bản transformers
    return (emb / emb.norm(dim=-1, keepdim=True)).numpy()


def clip_scores(db, q):
    """Cosine similarity (vector đã chuẩn hoá nên chỉ cần tích vô hướng)."""
    return db @ q


# ================= Chỉ mục + tìm kiếm =================
class Index:
    def __init__(self):
        paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
        if not paths:
            sys.exit("Chưa có ảnh trong images/ — chạy `python core.py prepare` trước.")
        datas = [p.read_bytes() for p in paths]
        imgs = [load_image(d) for d in datas]
        self.ids = [p.stem for p in paths]
        self.files = [p.name for p in paths]
        self.hashes = [hashlib.md5(d).hexdigest() for d in datas]
        print(f"Đang tính đặc trưng cho {len(paths)} ảnh...")
        self.feats = {"hist": np.stack([hist_feature(i) for i in imgs]), "clip": clip_feature(imgs)}

    def search(self, data, method, k=5):
        """data: bytes ảnh truy vấn -> [(id, file, score)] top-k.
        Nếu ảnh truy vấn trùng một ảnh trong tập (cùng nội dung file) thì loại nó khỏi kết quả."""
        img = load_image(data)
        if method == "hist":
            scores = hist_scores(self.feats["hist"], hist_feature(img))
        else:
            scores = clip_scores(self.feats["clip"], clip_feature([img])[0])
        qhash = hashlib.md5(data).hexdigest()
        ranked = [i for i in np.argsort(-scores) if self.hashes[i] != qhash]
        return [(self.ids[i], self.files[i], float(scores[i])) for i in ranked[:k]]


# ================= Chuẩn bị dữ liệu =================
def prepare():
    """Đầu vào (tự chọn tay từ dataset/archive/):
         images/<Lớp>/*.jpg   10 ảnh mỗi lớp
         queries/<Lớp>/*.jpg   2 ảnh mỗi lớp (KHÔNG trùng ảnh trong images/)
       Đầu ra: images/bicycle_01.jpg ..., queries/q01.jpg ..., sources.csv, ground_truth.csv
       Đáp án = các ảnh cùng lớp, được ghi TRƯỚC khi chạy so sánh."""
    sources, by_class = [], {}
    for cls_dir in sorted(d for d in IMG_DIR.iterdir() if d.is_dir()):
        cls = cls_dir.name.lower()
        for n, f in enumerate(sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS), 1):
            iid = f"{cls}_{n:02d}"
            shutil.move(f, IMG_DIR / f"{iid}{f.suffix.lower()}")
            sources.append([iid, cls, f"{cls_dir.name}/{f.name}", DATASET])
            by_class.setdefault(cls, []).append(iid)
        cls_dir.rmdir()

    gt, q = [], 0
    for cls_dir in sorted(d for d in Q_DIR.iterdir() if d.is_dir()):
        cls = cls_dir.name.lower()
        for f in sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXTS):
            q += 1
            qid = f"q{q:02d}"
            shutil.move(f, Q_DIR / f"{qid}{f.suffix.lower()}")
            sources.append([qid, cls, f"{cls_dir.name}/{f.name}", DATASET])
            gt.append([f"{qid}{f.suffix.lower()}", cls, " ".join(by_class.get(cls, []))])
        cls_dir.rmdir()

    write_csv(ROOT / "sources.csv", ["id", "class", "original_file", "dataset"], sources)
    write_csv(ROOT / "ground_truth.csv", ["query", "class", "relevant"], gt)
    print(f"Xong: {len(sources) - q} ảnh tìm kiếm, {q} truy vấn.")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])


# ================= Đánh giá Hit@5 =================
def evaluate():
    with open(ROOT / "ground_truth.csv", encoding="utf-8") as f:
        gt = list(csv.DictReader(f))
    index = Index()
    hits = {m: 0 for m in METHODS}
    lines = ["| Truy vấn | Lớp | " + " | ".join(f"{n} – top 5 | Hit (số đúng)" for n in METHODS.values()) + " |",
             "|---|---|" + "---|---|" * len(METHODS)]
    for row in gt:
        relevant = set(row["relevant"].split())
        qpath = Q_DIR / row["query"] if (Q_DIR / row["query"]).exists() else IMG_DIR / row["query"]
        cells = [row["query"], row["class"]]
        for m in METHODS:
            top = [iid for iid, _, _ in index.search(qpath.read_bytes(), m, k=5)]
            n_ok = sum(i in relevant for i in top)
            hits[m] += n_ok > 0
            cells += [", ".join(f"**{i}**" if i in relevant else i for i in top), f"{'✓' if n_ok else '✗'} ({n_ok}/5)"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "**Hit@5** — " + " | ".join(f"{METHODS[m]}: **{hits[m]}/{len(gt)}**" for m in METHODS),
              "", "_ID in đậm = kết quả đúng (cùng lớp với truy vấn)._"]
    report = "\n".join(lines)
    (ROOT / "results.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
    else:
        print(evaluate())
