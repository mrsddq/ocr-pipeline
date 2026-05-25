from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import IAMLineDataset, collate_iam_lines
from models import CRNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config: str, xml_dir: str, lines_dir: str) -> Path:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    dataset = IAMLineDataset(xml_dir, lines_dir, cfg["data"]["charset"], int(cfg["data"]["image_height"]))
    loader = DataLoader(dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, collate_fn=collate_iam_lines)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=len(cfg["data"]["charset"]) + 1, hidden_size=int(cfg["model"]["lstm_hidden"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    loss_fn = torch.nn.CTCLoss(blank=int(cfg["training"]["ctc_blank_idx"]), zero_infinity=True)
    for _epoch in range(int(cfg["training"]["epochs"])):
        for batch in loader:
            logits = model(batch["images"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device), batch["input_lengths"].to(device), batch["label_lengths"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    output_dir = Path(cfg["logging"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "crnn.pt"
    torch.save(model.state_dict(), output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN OCR model on IAM lines.")
    parser.add_argument("--config", default="configs/ocr.yaml")
    parser.add_argument("--xml-dir", required=True)
    parser.add_argument("--lines-dir", required=True)
    args = parser.parse_args()
    print(main(args.config, args.xml_dir, args.lines_dir))
