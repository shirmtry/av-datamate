from pathlib import Path

import pytest
import yaml

from avdatamate.config import ConfigError, PipelineConfig
from avdatamate.taxonomy import CLASS_TO_IDX, CLASSES, resolve_class

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_YAML = REPO_ROOT / "configs" / "sources.yaml"


def test_taxonomy_order_is_stable():
    assert CLASSES == ("car", "truck", "bus", "pedestrian", "cyclist", "traffic_light", "traffic_sign")
    assert [CLASS_TO_IDX[c] for c in CLASSES] == list(range(7))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Car", "car"),
        ("  VAN ", "car"),
        ("person", "pedestrian"),
        ("2-wheeler", "cyclist"),
        ("Traffic Sign", "traffic_sign"),
        ("traffic light", "traffic_light"),
        ("bicycle", "cyclist"),
        ("building", None),
        ("lorry", "truck"),
    ],
)
def test_resolve_class(raw, expected):
    assert resolve_class(raw) == expected


def test_shipped_sources_yaml_is_valid():
    cfg = PipelineConfig.load(SOURCES_YAML)
    assert len(cfg.sources) >= 2
    for source in cfg.sources:
        assert source.license in {"cc0-1.0", "cc-by-4.0", "cc-by-3.0", "mit", "apache-2.0", "odc-by"}, source.license
        assert source.id.count("/") == 1
        assert source.pretty_name
        assert source.upstream_repo.startswith("https://huggingface.co/")
        assert set(source.class_map.values()) <= set(CLASSES)


def test_source_class_maps_target_canonical_classes():
    cfg = PipelineConfig.load(SOURCES_YAML)
    for source in cfg.sources:
        assert source.class_map, source.id
        for target in source.class_map.values():
            assert resolve_class(target) == target


def test_bad_target_classes_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"target_classes": ["car"], "sources": [{"id": "a/b", "class_map": {}}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        PipelineConfig.load(path)


def test_source_without_id_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"sources": [{"license": "mit"}]}), encoding="utf-8")
    with pytest.raises(ConfigError):
        PipelineConfig.load(path)
