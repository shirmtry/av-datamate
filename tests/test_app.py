import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_app():
    spec = importlib.util.spec_from_file_location("avdatamate_app", REPO_ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return load_app()


@pytest.fixture
def mini_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            name = f"{split}_{i}"
            Image.new("RGB", (200, 120), (90, 120, 150)).save(root / "images" / split / f"{name}.jpg")
            (root / "labels" / split / f"{name}.txt").write_text(
                "0 0.5 0.5 0.2 0.3\n6 0.1 0.9 0.05 0.05\n", encoding="utf-8"
            )
    rows = []
    for split in ("train", "val"):
        for i in range(2):
            rows.append(
                {
                    "split": split,
                    "uid": f"{split}_{i}",
                    "source": "fake/x",
                    "image": f"dataset/images/{split}/{split}_{i}.jpg",
                    "label": f"dataset/labels/{split}/{split}_{i}.txt",
                    "n_boxes": 2,
                    "class_counts": {"car": 1, "traffic_sign": 1},
                }
            )
    (root / "splits.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


def test_label_to_boxes_uses_canonical_ids(app):
    boxes = app.label_to_boxes("0 0.5 0.5 0.2 0.3\n6 0.1 0.2 0.05 0.05\n99 0 0 1 1\ngarbage\n")
    assert [b["cls"] for b in boxes] == ["car", "traffic_sign"]
    assert boxes[0]["w"] == pytest.approx(0.2)


def test_annotate_draws_one_rectangle_per_box(app):
    image = Image.new("RGB", (200, 100), "white")
    boxes = [
        {"cls": "car", "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2},
        {"cls": "pedestrian", "cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1},
    ]
    out = app.annotate(image, boxes)
    assert out.size == image.size
    colors = {px[:3] for px in out.getdata()}
    assert any(c != (255, 255, 255) for c in colors)
    assert app.CLASS_COLORS["car"] in colors


def test_stats_table_from_splits_index(app, mini_dataset, monkeypatch):
    monkeypatch.setattr(app, "LOCAL_OUT", mini_dataset.parent)
    rows = app.stats_table(mini_dataset)
    assert rows[0] == ["class", "train", "val", "test", "total"]
    car = next(r for r in rows if r[0] == "car")
    assert car[1:4] == ["2", "2", "0"]
    images_row = next(r for r in rows if r[0] == "images")
    assert images_row[4] == "4"


def test_sample_gallery_returns_annotated_images(app, mini_dataset):
    gallery = app.sample_gallery(mini_dataset, "train", "car", 10)
    assert len(gallery) == 2
    image, caption = gallery[0]
    assert isinstance(image, Image.Image) and "boxes" in caption
    assert app.sample_gallery(mini_dataset, "test", "all", 10) == []


def test_load_index_falls_back_to_directory_scan(app, mini_dataset):
    (mini_dataset / "splits.jsonl").unlink()
    rows = app.load_index(mini_dataset)
    assert len(rows) == 4
    assert all(len(app.row_boxes(mini_dataset, r)) == 2 for r in rows)


def test_build_ui_requires_gradio(app):
    pytest.importorskip("gradio")
    assert callable(app.build_ui)
