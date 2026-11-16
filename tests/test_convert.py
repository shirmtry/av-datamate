
import yaml
from conftest import load_jsonl

from avdatamate.config import Layout, PipelineConfig
from avdatamate.convert import convert, data_yaml, yolo_label_text
from avdatamate.ingest import ingest
from avdatamate.taxonomy import CLASS_TO_IDX, CLASSES
from avdatamate.validate import validate


def pipeline_up_to_convert(config: PipelineConfig, layout: Layout, downloader):
    ingest(config, layout, 100, downloader, logger=lambda m: None)
    validate(config, layout, logger=lambda m: None)
    return convert(config, layout, logger=lambda m: None)


def test_convert_writes_yolo_files_and_data_yaml(config, layout, downloader):
    result = pipeline_up_to_convert(config, layout, downloader)
    index = load_jsonl(layout.converted_index)
    assert result.images == len(index) == 6
    doc = yaml.safe_load(layout.converted_data_yaml.read_text(encoding="utf-8"))
    assert doc["nc"] == len(CLASSES)
    assert doc["names"] == list(CLASSES)
    for row in index:
        image = layout.abs(row["image"])
        label = layout.abs(row["label"])
        assert image.exists() and label.exists()
        lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines) == row["n_boxes"]
        for line in lines:
            parts = line.split()
            assert int(parts[0]) in CLASS_TO_IDX.values()
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:5])
            assert len(parts[1].split(".")[1]) == 6


def test_convert_maps_only_canonical_classes(config, layout, downloader):
    result = pipeline_up_to_convert(config, layout, downloader)
    assert set(result.per_class) <= set(CLASSES)
    assert result.per_class["car"] >= 4
    assert sum(result.per_class.values()) == result.boxes
    assert result.boxes == sum(r["n_boxes"] for r in load_jsonl(layout.converted_index))


def test_yolo_label_text_roundtrip():
    boxes = [{"cls": "traffic_light", "cx": 0.5, "cy": 0.25, "w": 0.1, "h": 0.2}]
    idx, cx, cy, w, h = yolo_label_text(boxes).split()
    assert int(idx) == CLASS_TO_IDX["traffic_light"]
    assert (float(cx), float(cy), float(w), float(h)) == (0.5, 0.25, 0.1, 0.2)


def test_data_yaml_points_at_given_dirs():
    doc = yaml.safe_load(data_yaml({"train": "images/train", "val": "images/val"}))
    assert doc["train"] == "images/train"
    assert doc["path"] == "."
    assert doc["nc"] == 7
