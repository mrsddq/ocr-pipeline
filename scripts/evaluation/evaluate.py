"""Compute CER / WER for OCR predictions."""
import argparse
from pathlib import Path

import editdistance


def cer(prediction, reference):
    return editdistance.eval(prediction, reference) / max(len(reference), 1)


def wer(prediction, reference):
    return editdistance.eval(prediction.split(), reference.split()) / max(len(reference.split()), 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["tesseract", "crnn"], default="tesseract")
    p.add_argument("--checkpoint")
    a = p.parse_args()
    gt_dir, pred_dir = Path("data/ground_truth"), Path("outputs")
    cers, wers = [], []
    for gt in gt_dir.glob("*.txt"):
        pred_f = pred_dir / gt.name.replace(".txt", "_ocr.txt")
        if not pred_f.exists():
            continue
        ref, pred = gt.read_text().strip(), pred_f.read_text().strip()
        cers.append(cer(pred, ref))
        wers.append(wer(pred, ref))
    if cers:
        print(f"CER: {sum(cers) / len(cers):.4f}")
        print(f"WER: {sum(wers) / len(wers):.4f}")
    else:
        print("No pairs found - run infer.py first.")
