# Ablation Plan

| Experiment | Variable | Fixed Controls | Metric |
|---|---|---|---|
| baseline | raw image OCR | dataset split | CER/WER |
| denoising | denoise strength | engine, dataset | CER/WER |
| thresholding | Otsu vs adaptive | engine, dataset | CER/WER |
| contrast | CLAHE on/off | engine, dataset | CER/WER |
| engine | Tesseract vs EasyOCR/PaddleOCR | input images | CER/WER and latency |

Each ablation should save config, command, metrics CSV, and example error analysis.
