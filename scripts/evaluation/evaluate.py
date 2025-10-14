"""Compute CER / WER."""
import argparse
from pathlib import Path
import editdistance

def cer(p,r): return editdistance.eval(p, r) / max(len(r), 1)
def wer(p,r): return editdistance.eval(p.split(), r.split()) / max(len(r.split()), 1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    a = p.parse_args()
    gt_dir, pred_dir = Path("data/ground_truth"), Path("outputs")
    cers, wers = [], []
    for gt in gt_dir.glob("*.txt"):
        pred_f = pred_dir / gt.name.replace(".txt", "_ocr.txt")
        if not pred_f.exists(): continue
        ref, pred = gt.read_text().strip(), pred_f.read_text().strip()
        cers.append(cer(pred, ref)); wers.append(wer(pred, ref))
    if cers:
        print(f"Char accuracy: {1 - sum(cers)/len(cers):.4f}")
        print(f"Word accuracy: {1 - sum(wers)/len(wers):.4f}")
    else:
        print("No pairs found — run infer.py first.")
