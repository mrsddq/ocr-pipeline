import json
import pytest
import torch
import yaml
from PIL import Image
from scripts.inference.infer import ctc_decode, main
from scripts.evaluation.evaluate import evaluate, cer, wer
from models import CRNN
from data import IAMLineDataset


def test_ctc_repeats_are_separated_by_blank():
    assert ctc_decode([0, 1, 1, 0, 1, 2, 2, 0], "AB") == "AAB"
    with pytest.raises(ValueError):
        ctc_decode([3], "AB")


def test_corpus_weighting_and_missing_predictions(tmp_path):
    refs, preds = tmp_path / "refs", tmp_path / "preds"
    refs.mkdir(); preds.mkdir()
    (refs / "a.txt").write_text("a")
    (refs / "b.txt").write_text("b" * 9)
    (preds / "a_ocr.txt").write_text("x")
    with pytest.raises(FileNotFoundError):
        evaluate(refs, preds)
    (preds / "b_ocr.txt").write_text("b" * 9)
    report = evaluate(refs, preds)
    assert report["cer"] == pytest.approx(0.1)
    assert report["wer"] == pytest.approx(0.5)
    assert cer("abc", "") == 3
    assert wer("one two", "one") == 1


def test_crnn_checkpoint_inference_is_real_and_offline(tmp_path):
    torch.set_num_threads(1)
    cfg = {"data": {"charset": "AB", "image_height": 16}, "model": {"lstm_hidden": 8}}
    model = CRNN(3, hidden_size=8)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-20., 20., -20.]))
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    image = tmp_path / "line.png"
    Image.new("L", (40, 16), 255).save(image)
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = main(image, tmp_path / "out", "crnn", checkpoint, config)
    assert output.read_text() == "A"


def test_iam_rejects_silently_dropped_unknown_characters(tmp_path):
    xml, images = tmp_path / "xml", tmp_path / "images"
    xml.mkdir(); images.mkdir()
    (xml / "a.xml").write_text('<root><line id="a01-001-00" text="AB?" /></root>')
    Image.new("L", (50, 16), 255).save(images / "a01-001-00.png")
    with pytest.raises(ValueError, match="Unsupported characters"):
        IAMLineDataset(xml, images, "AB", 16)
