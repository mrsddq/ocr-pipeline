"""Infer OCR on a single image."""
import argparse, yaml
from pathlib import Path
import pytesseract
import sys; sys.path.insert(0, ".")
from scripts.preprocessing.preprocess import preprocess

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs/")
    a = p.parse_args()
    with open("configs/ocr.yaml") as f:
        cfg = yaml.safe_load(f)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    processed = preprocess(a.input, cfg["preprocessing"])
    text = pytesseract.image_to_string(processed)
    out = Path(a.output) / (Path(a.input).stem + "_ocr.txt")
    out.write_text(text)
    print(f"Saved: {out}")
