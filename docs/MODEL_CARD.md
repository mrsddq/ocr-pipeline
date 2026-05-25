# Model Card: OCR Pipeline

## Dataset
Target dataset: IAM Handwriting Database lines. Fallback dataset: MJSynth/Synth90k for synthetic word recognition.

## Model
Baseline engine: Tesseract. Main model: CRNN with CNN feature extraction, BiLSTM sequence modeling, and CTC loss.

## Evaluation
Primary metrics: Character Error Rate and Word Error Rate.

## Limitations
Handwriting quality, skew, and domain shift can strongly affect recognition. CRNN decoding requires a trained checkpoint.

## Ethical Considerations
OCR may expose private information from documents. Use only documents you have permission to process.
