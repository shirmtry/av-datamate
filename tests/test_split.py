from pathlib import Path

import pytest
from conftest import add_clip, load_jsonl

from avdatamate.config import Layout, PipelineConfig
from avdatamate.convert import convert
from avdatamate.ingest import ingest
from avdatamate.split import RATIOS, assign_groups, split
from avdatamate.validate import validate


@pytest.fixture
def converted(config: PipelineConfig, layout: Layout, downloader, fake_repo):
    for i in range(8):
        add_clip(fake_repo, f"MVI_{4000 + i}", n=3)
    build = ingest(config, layout, 1000, downloader, logger=lambda m: None)
    assert build.selected == 35
    validate(config, layout, logger=lambda m: None)
    convert(config, layout, logger=lambda m: None)
    assert len(load_jsonl(layout.converted_index)) == 30
    return config, layout


def test_split_keeps_groups_whole(converted):
    config, layout = converted
    result = split(config, layout, seed=1337, logger=lambda m: None)
    rows = load_jsonl(layout.splits_index)
    assert len(rows) == sum(result.counts.values()) == 30
    assert result.groups == 10
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(f"{row['source']}::{row['group']}", set()).add(row["split"])
    assert all(len(s) == 1 for s in by_group.values()), "a scene group leaked across splits"


def test_split_removes_stale_files_from_earlier_runs(converted):
    config, layout = converted
    split(config, layout, seed=1337, logger=lambda m: None)
    stale = layout.dataset_images / "train" / "stale_leftover.jpg"
    stale.write_bytes(b"not from this run")
    split(config, layout, seed=99, logger=lambda m: None)
    assert not stale.exists()
    rows = load_jsonl(layout.splits_index)
    on_disk = [p.name for p in layout.dataset_images.rglob("*.jpg")]
    assert sorted(on_disk) == sorted(Path(r["image"]).name for r in rows)
    labels = [p.name for p in layout.dataset_labels.rglob("*.txt")]
    assert sorted(labels) == sorted(Path(r["label"]).name for r in rows)


def test_split_fills_every_split(converted):
    config, layout = converted
    result = split(config, layout, seed=7, logger=lambda m: None)
    assert all(result.counts[p] > 0 for p in RATIOS)
    assert result.counts["train"] >= result.counts["val"]
    assert result.counts["train"] >= result.counts["test"]
    for part in RATIOS:
        assert (layout.dataset_images / part).exists()
        assert list((layout.dataset_labels / part).glob("*.txt"))


def test_split_is_deterministic_for_a_seed(converted):
    config, layout = converted
    split(config, layout, seed=1337, logger=lambda m: None)
    first = {r["uid"]: r["split"] for r in load_jsonl(layout.splits_index)}
    split(config, layout, seed=1337, logger=lambda m: None)
    second = {r["uid"]: r["split"] for r in load_jsonl(layout.splits_index)}
    assert first == second
    other = assign_groups(
        {f"a::{g}": [{}] for g in ("x", "y", "z")}, seed=99
    )
    assert len(set(other.values())) >= 1


def test_split_needs_converted_index(config, layout):
    with pytest.raises(RuntimeError):
        split(config, layout, 1, logger=lambda m: None)


def test_split_data_yaml_lists_split_dirs(converted):
    import yaml

    config, layout = converted
    split(config, layout, seed=3, logger=lambda m: None)
    doc = yaml.safe_load(layout.dataset_data_yaml.read_text(encoding="utf-8"))
    assert doc["train"] == "images/train"
    assert doc["val"] == "images/val"
    assert doc["test"] == "images/test"
    assert doc["nc"] == 7
