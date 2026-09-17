"""Tesseract or CRNN inference, with the same CRNN preprocessing as training."""
import argparse
from pathlib import Path
import yaml


def ctc_decode(ids, charset, blank=0):
    result, previous = [], None
    for token in ids:
        token = int(token)
        if token != blank and token != previous:
            if not 1 <= token <= len(charset):
                raise ValueError("CTC token lies outside charset")
            result.append(charset[token - 1])
        previous = token
    return "".join(result)


def infer_crnn(image_path, checkpoint, cfg):
    import torch
    from PIL import Image
    from models import CRNN
    charset = cfg["data"]["charset"]
    model = CRNN(num_classes=len(charset) + 1, hidden_size=int(cfg["model"]["lstm_hidden"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with Image.open(image_path) as source:
        image = source.convert("L")
    height = int(cfg["data"]["image_height"])
    width = max(4, round(image.width * height / image.height))
    image = image.resize((width, height))
    values = torch.tensor(list(image.getdata()), dtype=torch.float32).view(1, 1, height, width) / 255.0
    with torch.inference_mode():
        ids = model(values).argmax(-1)[:, 0].tolist()
    return ctc_decode(ids, charset)


def main(input_path, output, engine="tesseract", checkpoint=None, config="configs/ocr.yaml"):
    if not Path(input_path).is_file():
        raise FileNotFoundError(input_path)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if engine == "crnn":
        if not checkpoint:
            raise ValueError("--checkpoint is required when --engine crnn")
        text = infer_crnn(input_path, checkpoint, cfg)
    elif engine == "tesseract":
        import pytesseract
        from scripts.preprocessing.preprocess import preprocess
        text = pytesseract.image_to_string(preprocess(input_path, cfg["preprocessing"]))
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    destination = Path(output) / (Path(input_path).stem + "_ocr.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="outputs")
    p.add_argument("--engine", choices=["tesseract", "crnn"], default="tesseract")
    p.add_argument("--checkpoint")
    p.add_argument("--config", default="configs/ocr.yaml")
    a = p.parse_args()
    print(main(a.input, a.output, a.engine, a.checkpoint, a.config))
