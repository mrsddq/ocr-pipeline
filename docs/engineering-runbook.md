# OCR engineering runbook

This repository supports image preprocessing, Tesseract inference, CRNN training
and greedy CTC decoding, and paired-text CER/WER evaluation. Use the
[README setup](../README.md#setup), [inference](../README.md#inference),
[evaluation](../README.md#evaluation), and
[CRNN workflow](../README.md#implemented-engineering-checks). Tesseract is a host
executable dependency for that engine; it is not needed by the CPU test suite.

## Local verification

Run from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Dependency installation needs package-network access. Once installed, the test
suite runs on CPU with generated fixtures and does not download model weights or
datasets. On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell
and run `python -m pytest -q`.

Tests execute a controlled CRNN checkpoint and cover CTC blank/repeat behavior,
configured recurrent depth, IAM sample handling, corpus-weighted CER/WER, and
missing-prediction errors. They establish execution behavior, not handwriting or
multilingual recognition accuracy.

## Data and artifact contract

- Acquire IAM or other datasets under their access/license terms. IAM images may
  use flat or official nested paths. Missing images and unsupported characters
  fail explicitly; update the intended charset instead of dropping labels.
- CRNN training uses aspect-preserving grayscale line images. A target that cannot
  align after the CNN width reduction is an error, not a zero-loss example.
- The checkpoint is a raw state dict. Preserve the exact YAML charset, recurrent
  depth and other model dimensions with it; inference must use matching settings.
- Validation-based model selection remains future work. Keep training and held-out
  evaluation records separate when running real experiments.
- Every reference `<stem>.txt` needs a prediction `<stem>_ocr.txt`. Evaluation strips
  outer whitespace, preserves case, and reports corpus error counts/denominators
  plus per-sample errors. Insertion-heavy CER/WER can exceed one.
- Keep private document images, transcriptions, generated text and checkpoints
  outside git. Publish only permissioned examples and reproducible metrics.

The README records supported commands and current limitations. A successful
synthetic checkpoint test is not a trained OCR accuracy result.
