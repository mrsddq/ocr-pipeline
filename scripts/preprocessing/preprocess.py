"""Preprocessing: denoise -> CLAHE -> Otsu -> Sobel."""
import argparse, cv2, numpy as np
from pathlib import Path

def preprocess(img_path, cfg):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.fastNlMeansDenoising(img, h=cfg["denoise_h"])
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=tuple(cfg["clahe_grid"]))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=cfg["sobel_ksize"])
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=cfg["sobel_ksize"])
    edges = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    return cv2.addWeighted(img, 0.7, edges, 0.3, 0)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/")
    p.add_argument("--output", default="data/preprocessed/")
    a = p.parse_args()
    import yaml
    with open("configs/ocr.yaml") as f:
        cfg = yaml.safe_load(f)["preprocessing"]
    inp, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    images = list(inp.glob("*.png")) + list(inp.glob("*.jpg"))
    for p2 in images:
        cv2.imwrite(str(out / p2.name), preprocess(str(p2), cfg))
    print(f"Done — {len(images)} images")
