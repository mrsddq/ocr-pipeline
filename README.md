# OCR Pipeline

[![CI](https://github.com/mrsddq/ocr-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsddq/ocr-pipeline/actions/workflows/ci.yml)

Portfolio-ready OCR pipeline for document image preprocessing, inference, and evaluation.

The repository is designed around a practical OCR workflow: clean noisy scans, run OCR, compare extracted text against references, and document failure cases. It does not include private datasets or unverified accuracy claims.

## Highlights

- OpenCV preprocessing pipeline
- Tesseract-compatible inference entry point
- Evaluation script structure for CER/WER-style reporting
- YAML configuration
- Clear artifact plan for screenshots and error analysis

## Structure

```text
configs/
  ocr.yaml
docs/
  ABLATION_PLAN.md
  ARCHITECTURE_RATIONALE.md
  DEPLOYMENT_NOTES.md
  REPRODUCIBILITY.md
  RESULTS_TEMPLATE.md
scripts/
  preprocessing/preprocess.py
  inference/infer.py
  evaluation/evaluate.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tesseract OCR must also be installed on the host machine if using the Tesseract fallback path.

## Data Layout

```text
data/
  raw/
  preprocessed/
  ground_truth/
```

## Preprocess

```bash
python -m scripts.preprocessing.preprocess --input data/raw --output data/preprocessed
```

## Inference

```bash
python -m scripts.inference.infer --input data/raw/sample.png --output outputs/
```

## Evaluation

```bash
python -m scripts.evaluation.evaluate --references data/ground_truth --predictions outputs --output outputs/metrics/ocr.json
```

## Results

No verified public metrics are committed yet. Add a reproducible evaluation table after running on a public dataset such as IAM or ICDAR.

Research support docs:

- [Portfolio Evidence Plan](docs/PORTFOLIO_EVIDENCE.md)
- [Reproducibility Plan](docs/REPRODUCIBILITY.md)
- [Architecture Rationale](docs/ARCHITECTURE_RATIONALE.md)
- [Ablation Plan](docs/ABLATION_PLAN.md)
- [Deployment Notes](docs/DEPLOYMENT_NOTES.md)

`outputs/metrics/smoke_test_results.csv` is a schema artifact only, not a benchmark.

Recommended artifacts:

- `assets/preprocessing-strip.png`
- `assets/ocr-output.png`
- `assets/error-analysis.png`
- `assets/metrics-summary.png`

## Limitations

- No dataset is included.
- CRNN training and greedy decoding are implemented; no trained model is supplied.
- Handwritten, multilingual, and low-resolution documents need separate evaluation.

## Implemented engineering checks

`python -m pytest -q` runs CPU/offline tests using `requirements-test.txt`.
The tests execute a CRNN checkpoint through image loading and CTC decoding, check
blank/repeat semantics, corpus-weighted error rates, and malformed datasets.
These synthetic tests validate execution, not handwriting recognition accuracy.

```bash
python -m scripts.train --config configs/ocr.yaml --xml-dir data/iam/xml --lines-dir data/iam/lines
python -m scripts.inference.infer --engine crnn --checkpoint outputs/crnn.pt --input data/sample.png --output outputs
```

Use the actual `logging.checkpoint_dir` from your configuration. IAM images can
use flat or official nested line directories. Acquire data under its own access
and license terms. Unknown characters and missing images fail explicitly; extend
the charset intentionally instead of silently dropping labels. CRNN targets that
cannot align at the CNN's width reduction fail instead of becoming zero-loss
training samples. Training saves a raw state dict: retain the exact YAML charset
and model dimensions with it. Validation-based checkpoint selection is future work.

Evaluation requires every reference to have a `<stem>_ocr.txt` prediction, strips
outer whitespace, is case-sensitive, and reports corpus CER/WER with numerator
and denominator counts plus per-sample errors. Rates can exceed 1 for insertions.
This avoids the prior missing-prediction skip and unweighted per-line average.
