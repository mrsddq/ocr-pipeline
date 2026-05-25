"""OCR datasets."""

from .iam_dataset import IAMLineDataset, collate_iam_lines

__all__ = ["IAMLineDataset", "collate_iam_lines"]
