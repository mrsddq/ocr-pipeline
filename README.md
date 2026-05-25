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
.venv\Scripts\activate
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
python -m scripts.evaluation.evaluate --checkpoint outputs/best_model.pt --data data/preprocessed
```

## Results

No verified public metrics are committed yet. Add a reproducible evaluation table after running on a public dataset such as IAM or ICDAR.

Research support docs:

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
- Deep OCR model training is an extension point.
- Handwritten, multilingual, and low-resolution documents need separate evaluation.
