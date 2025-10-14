# OCR Pipeline

End-to-end OCR system combining classical image preprocessing (OpenCV) with deep sequence modelling (CNN + LSTM) and Tesseract fallback. Achieves >92% character-level accuracy on document images.

## Results

| Metric | Value | Notes |
|---|---|---|
| Character Accuracy | >92% | On held-out test set |
| Word Accuracy | _add_ | After re-evaluation |
| Field-level Accuracy | _add_ | Form/invoice documents |
| CER | _add_ | Character Error Rate |
| WER | _add_ | Word Error Rate |

## Pipeline

```
Raw document image
  └─ Denoising          (OpenCV fastNlMeans)
       └─ Histogram EQ  (CLAHE)
            └─ Thresholding (Otsu adaptive)
                 └─ Sobel edge enhancement
                      └─ CNN feature extractor
                           └─ BiLSTM sequence model
                                └─ CTC decode  →  text output
```

Tesseract is used as a fallback for low-confidence CNN-LSTM outputs.

## Related Publication

> Enhanced Document Image Pre-processing for OCR Accuracy — _[add link/DOI]_

## Quickstart

```bash
git clone https://github.com/your-username/ocr-pipeline
cd ocr-pipeline
pip install -r requirements.txt
```

## Data

Supports any document image dataset. Tested on:

- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [ICDAR 2019 Competition](https://rrc.cvc.uab.es/)

```
data/
  raw/             ← original document scans
  preprocessed/    ← output of preprocessing pipeline
  ground_truth/    ← .txt files with reference transcriptions
```

## Preprocessing

```bash
python scripts/preprocessing/preprocess.py --input data/raw/ --output data/preprocessed/
```

Applies: denoise → CLAHE → Otsu threshold → Sobel enhancement.

## Training

```bash
python scripts/train.py --config configs/ocr.yaml
```

## Evaluation

```bash
python scripts/evaluation/evaluate.py --checkpoint outputs/best_model.pt --data data/preprocessed/
```

## Inference

```bash
python scripts/inference/infer.py --input data/raw/sample.png --output outputs/
```

## Sample Outputs

| File | Contents |
|---|---|
| `assets/01_preprocessing_strip.png` | 5-stage preprocessing pipeline visualised |
| `assets/02_ocr_output.png` | Document image + extracted text side-by-side |
| `assets/03_error_analysis.png` | Failure categories: blur, skew, low contrast, etc. |
| `assets/04_accuracy_metrics.png` | Terminal output showing char/word accuracy |

## Limitations

- Performance degrades on handwritten text (model tuned for printed documents)
- Multilingual documents not tested beyond Latin scripts
- Tesseract fallback adds latency on long documents

## Environment

```
Python 3.10
torch==2.1.0
opencv-python==4.8.1.78
pytesseract==0.3.10
Tesseract-OCR >= 5.0
```
