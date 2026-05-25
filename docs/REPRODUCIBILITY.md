# Reproducibility Plan

## Environment

- Python: 3.10
- Dependencies: pinned in `requirements.txt`
- Config: `configs/ocr.yaml`
- External engine: record Tesseract version when used

## Dataset Contract

For each run, record dataset name/version, document type, language/script, preprocessing settings, ground-truth source, and checksum or DVC hash.

## Run Order

1. Preprocess raw documents.
2. Run OCR inference.
3. Compare extracted text with ground truth.
4. Report CER/WER and common error categories.
5. Save small example outputs under `assets/`.

`outputs/metrics/smoke_test_results.csv` is a schema example only. It is not OCR accuracy.
