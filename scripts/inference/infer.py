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
    p.add_argument("--engine", choices=["tesseract", "crnn"], default="tesseract")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    with open("configs/ocr.yaml") as f:
        cfg = yaml.safe_load(f)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    processed = preprocess(a.input, cfg["preprocessing"])
    if a.engine == "tesseract":
        text = pytesseract.image_to_string(processed)
    else:
        if not a.checkpoint:
            raise ValueError("--checkpoint is required when --engine crnn")
        text = "[CRNN decoding scaffold] load checkpoint and run CTC decode here."
    out = Path(a.output) / (Path(a.input).stem + "_ocr.txt")
    out.write_text(text)
    print(f"Saved: {out}")
