# Deployment Notes

Use Streamlit or Gradio for a demo:

- upload image/PDF
- preview preprocessed image
- show extracted text
- show optional confidence/error notes

Serving notes:

- limit file size and page count
- strip metadata from uploaded files
- do not store private documents by default
- record OCR engine version for reproducibility
