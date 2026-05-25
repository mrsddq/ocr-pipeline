from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset


class IAMLineDataset(Dataset):
    def __init__(self, xml_dir: str | Path, lines_dir: str | Path, charset: str, image_height: int = 64) -> None:
        self.xml_dir = Path(xml_dir)
        self.lines_dir = Path(lines_dir)
        self.charset = charset
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
        self.image_height = image_height
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, text = self.samples[index]
        image = Image.open(image_path).convert("L")
        width = max(1, round(image.width * self.image_height / image.height))
        image = image.resize((width, self.image_height))
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(self.image_height, width) / 255.0
        label = torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)
        return {"image": tensor.unsqueeze(0), "label": label, "text": text}

    def _load_samples(self) -> list[tuple[Path, str]]:
        samples = []
        for xml_path in self.xml_dir.glob("*.xml"):
            root = ET.parse(xml_path).getroot()
            for line in root.iter("line"):
                line_id = line.attrib["id"]
                text = line.attrib.get("text", "")
                image_path = self.lines_dir / f"{line_id}.png"
                if image_path.exists() and text:
                    samples.append((image_path, text))
        return samples


def collate_iam_lines(batch):
    max_width = max(item["image"].shape[-1] for item in batch)
    images, labels, label_lengths, input_lengths = [], [], [], []
    for item in batch:
        image = item["image"]
        padded = torch.zeros((1, image.shape[1], max_width), dtype=image.dtype)
        padded[:, :, : image.shape[-1]] = image
        images.append(padded)
        labels.append(item["label"])
        label_lengths.append(len(item["label"]))
        input_lengths.append(max(1, image.shape[-1] // 4))
    return {
        "images": torch.stack(images),
        "labels": torch.cat(labels),
        "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
    }
