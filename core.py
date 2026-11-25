"""CS232 HT2 - Core: 2 cách biểu diễn ảnh, tìm top-k, đánh giá Hit@5.

Cài đặt : pip install torch transformers pillow numpy
Dữ liệu : images/01.jpg ... 50.jpg (tên file = ID), ground_truth.csv, ảnh truy vấn ngoài tập để trong queries/
Đánh giá: python core.py  -> in bảng + ghi results.md
"""
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
IMG_DIR, Q_DIR = ROOT / "images", ROOT / "queries"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLIP_NAME = "openai/clip-vit-base-patch32"
METHODS = {"hist": "Histogram màu HSV", "clip": "CLIP embedding"}


def load_ground_truth():
    """ground_truth.csv: query,relevant  (vd: q01.jpg,03 17 22) — GHI TRƯỚC khi chạy so sánh.
    query là tên file trong queries/ hoặc images/ (ảnh trong tập sẽ tự bị loại khỏi kết quả)."""
    with open(ROOT / "ground_truth.csv", encoding="utf-8") as f:
        return {r["query"]: r["relevant"].split() for r in csv.DictReader(f)}


# ---------- Phương pháp 1: histogram màu HSV 8x4x4, so bằng histogram intersection ----------
def hist_feature(img, bins=(8, 4, 4)):
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    h, _ = np.histogramdd(hsv, bins=bins, range=[(0, 256)] * 3)
    return h.flatten() / h.sum()


# ---------- Phương pháp 2: embedding CLIP ViT-B/32, so bằng cosine ----------
_clip = None


def clip_feature(imgs):
    global _clip
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if _clip is None:
        _clip = CLIPModel.from_pretrained(CLIP_NAME).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)
    model, proc = _clip
    with torch.no_grad():
        emb = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
    return (emb / emb.norm(dim=-1, keepdim=True)).numpy()


# ---------- Index ----------
def _load(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


class Index:
    def __init__(self):
        paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS)
        datas = [p.read_bytes() for p in paths]
        imgs = [_load(d) for d in datas]
        self.files = [p.name for p in paths]
        self.ids = [p.stem for p in paths]
        self.hashes = [hashlib.md5(d).hexdigest() for d in datas]
        self.feats = {"hist": np.stack([hist_feature(i) for i in imgs]), "clip": clip_feature(imgs)}

    def search(self, data, method="clip", k=5):
        """data: bytes của ảnh truy vấn. Trả về [(id, file, score)] top-k, loại chính ảnh truy vấn nếu có trong tập."""
        img = _load(data)
        if method == "hist":
            scores = np.minimum(self.feats["hist"], hist_feature(img)).sum(axis=1)
        else:
            scores = self.feats["clip"] @ clip_feature([img])[0]
        qhash = hashlib.md5(data).hexdigest()
        order = [i for i in np.argsort(-scores) if self.hashes[i] != qhash]
        return [(self.ids[i], self.files[i], float(scores[i])) for i in order[:k]]


# ---------- Đánh giá Hit@5 ----------
def evaluate(index=None):
    index = index or Index()
    gt = load_ground_truth()
    hits = {m: 0 for m in METHODS}
    lines = ["| Truy vấn | ID đúng | " + " | ".join(f"{n} top-5 | Hit" for n in METHODS.values()) + " |",
             "|---|---|" + "---|---|" * len(METHODS)]
    for q, relevant in gt.items():
        data = (Q_DIR / q if (Q_DIR / q).exists() else IMG_DIR / q).read_bytes()
        cells = [q, " ".join(relevant)]
        for m in METHODS:
            top = [iid for iid, _, _ in index.search(data, m, k=5)]
            hit = any(i in relevant for i in top)
            hits[m] += hit
            cells += [" ".join(f"**{i}**" if i in relevant else i for i in top), "✓" if hit else "✗"]
        lines.append("| " + " | ".join(cells) + " |")
    n = len(gt)
    lines += ["", "**Hit@5** — " + " | ".join(f"{METHODS[m]}: {hits[m]}/{n}" for m in METHODS),
              "", "_(ID in đậm = kết quả đúng)_"]
    report = "\n".join(lines)
    (ROOT / "results.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(evaluate())
