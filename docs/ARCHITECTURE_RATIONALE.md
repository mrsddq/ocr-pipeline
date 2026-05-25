# Architecture Rationale

OCR quality depends on the whole document pipeline, not only the recognizer:

```text
raw scan -> denoise/contrast/binarize -> OCR engine -> cleanup -> CER/WER evaluation
```

The current inference path is Tesseract-compatible because it is accessible and easy to reproduce. Deep OCR models can be added later behind the same inference interface.

Upgrade path:

- add layout detection
- add spell/regex post-processing
- compare Tesseract, EasyOCR, and PaddleOCR
- add PDF page extraction
