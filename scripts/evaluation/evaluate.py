"""Corpus CER/WER for exact reference/prediction pairs; no hidden missing-file skip."""
import argparse
import json
from pathlib import Path


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def cer(prediction, reference):
    return edit_distance(prediction, reference) / max(len(reference), 1)


def wer(prediction, reference):
    return edit_distance(prediction.split(), reference.split()) / max(len(reference.split()), 1)


def evaluate(reference_dir, prediction_dir):
    references = sorted(Path(reference_dir).glob("*.txt"))
    if not references:
        raise ValueError("No reference text files found")
    missing = [p.name for p in references if not (Path(prediction_dir) / f"{p.stem}_ocr.txt").is_file()]
    if missing:
        raise FileNotFoundError("Missing predictions: " + ", ".join(missing))
    chars = words = char_errors = word_errors = 0
    rows = []
    for path in references:
        reference = path.read_text(encoding="utf-8").strip()
        prediction = (Path(prediction_dir) / f"{path.stem}_ocr.txt").read_text(encoding="utf-8").strip()
        ce, we = edit_distance(prediction, reference), edit_distance(prediction.split(), reference.split())
        chars += len(reference)
        words += len(reference.split())
        char_errors += ce
        word_errors += we
        rows.append({"id": path.stem, "character_errors": ce, "word_errors": we, "reference_characters": len(reference), "reference_words": len(reference.split())})
    if not chars or not words:
        raise ValueError("Corpus must contain nonempty reference characters and words")
    return {"samples": len(rows), "cer": char_errors / chars, "wer": word_errors / words,
            "character_errors": char_errors, "reference_characters": chars,
            "word_errors": word_errors, "reference_words": words, "per_sample": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--references", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default="outputs/metrics/ocr.json")
    a = p.parse_args()
    report = evaluate(a.references, a.predictions)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))
