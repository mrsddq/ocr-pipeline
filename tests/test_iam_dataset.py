import pytest
import torch
from PIL import Image

from data.iam_dataset import IAMLineDataset, collate_iam_lines


def test_dataset_indexes_valid_iam_line(tmp_path):
    xml_dir = tmp_path / "xml"
    lines_dir = tmp_path / "lines"
    xml_dir.mkdir()
    lines_dir.mkdir()
    (xml_dir / "sample.xml").write_text('<root><line id="a01-000u-00" text="HELLO" /></root>', encoding="utf-8")
    Image.new("L", (20, 10), color=255).save(lines_dir / "a01-000u-00.png")

    dataset = IAMLineDataset(xml_dir, lines_dir, charset="HELO", image_height=16)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape[1] == 16
    assert sample["label"].numel() == 5


def test_dataset_rejects_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="XML directory"):
        IAMLineDataset(tmp_path / "missing", tmp_path / "lines", charset="ABC")


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one"):
        collate_iam_lines([])


def test_collate_pads_variable_width_lines():
    batch = [
        {"image": torch.ones(1, 8, 3), "label": torch.tensor([1, 2]), "text": "AB"},
        {"image": torch.ones(1, 8, 5), "label": torch.tensor([1]), "text": "A"},
    ]

    collated = collate_iam_lines(batch)

    assert collated["images"].shape == (2, 1, 8, 5)
    assert collated["labels"].tolist() == [1, 2, 1]
