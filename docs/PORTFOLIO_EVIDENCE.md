# Portfolio Evidence Plan

This project should be shown as an OCR pipeline with preprocessing, inference, and error analysis. Do not claim OCR accuracy until a public or shareable dataset run is documented.

## Reproducible Demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_portfolio_contract.py"
python -m scripts.preprocessing.preprocess --input data/raw --output data/preprocessed
python -m scripts.inference.infer --input data/raw/sample.png --output outputs/
python -m scripts.evaluation.evaluate --checkpoint outputs/best_model.pt --data data/preprocessed
```

## Evidence To Capture

| Artifact | Portfolio Use |
|---|---|
| `assets/preprocessing-strip.png` | Shows raw, denoised, thresholded, and deskewed stages. |
| `assets/ocr-output.png` | Shows extracted text next to the source image. |
| `assets/error-analysis.png` | Shows common OCR failure cases. |
| `outputs/metrics/ocr_eval.csv` | Records CER, WER, exact match, and dataset split. |
| `docs/RESULTS.md` | Summarizes only verified OCR runs. |

## Demo Narrative

1. Start from a noisy document image.
2. Show each preprocessing stage and why it helps OCR.
3. Run inference and compare extracted text with ground truth.
4. Report CER/WER and discuss failure modes such as blur, handwriting, tables, and mixed languages.

## Evidence Checklist Before Pinning

- [ ] Public or synthetic document sample identified.
- [ ] Preprocessing strip added to `assets/`.
- [ ] Real CER/WER table added to `docs/RESULTS.md`.
- [ ] CI badge green on the latest commit.
- [ ] Tesseract/system dependency notes verified.
