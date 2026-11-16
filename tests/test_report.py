from conftest import add_clip, load_jsonl

from avdatamate.convert import convert
from avdatamate.ingest import ingest
from avdatamate.report import build, render_markdown, report
from avdatamate.split import split
from avdatamate.validate import validate


def test_report_writes_json_and_markdown(config, layout, downloader, fake_repo):
    add_clip(fake_repo, "MVI_3001", n=4)
    add_clip(fake_repo, "MVI_3002", n=4)
    ingest(config, layout, 100, downloader, logger=lambda m: None)
    validate(config, layout, logger=lambda m: None)
    convert(config, layout, logger=lambda m: None)
    split(config, layout, seed=11, logger=lambda m: None)
    data = report(config, layout, logger=lambda m: None)

    assert data["counts"]["published"] == 14
    assert data["counts"]["validated"] == len(load_jsonl(layout.validated)) == 14
    assert data["counts"]["boxes"] == sum(
        r["n_boxes"] for r in load_jsonl(layout.converted_index)
    )
    assert data["leakage"]["clean"] is True
    assert data["class_distribution"]["boxes"]["traffic_light"] == 0
    assert "traffic_light" in data["class_distribution"]["missing_classes"]
    assert data["image_sizes"]["min"] >= 32
    assert data["rejected_by_reason"]["corrupt_image"] == 1
    assert data["per_source"]["fake/traffic-mini"]["license"] == "mit"
    assert layout.report_json.exists() and layout.report_md.exists()


def test_markdown_mentions_splits_classes_and_sources(config, layout, downloader, fake_repo):
    add_clip(fake_repo, "MVI_4001", n=2)
    ingest(config, layout, 100, downloader, logger=lambda m: None)
    validate(config, layout, logger=lambda m: None)
    convert(config, layout, logger=lambda m: None)
    split(config, layout, seed=5, logger=lambda m: None)
    data = build(config, layout)
    text = render_markdown(data)
    for token in ("## Splits", "## Class distribution", "traffic_sign", "car", "fake/traffic-mini", "mit"):
        assert token in text
    assert "Leakage check: PASS" in text


def test_report_survives_empty_rejections(config, layout, downloader):
    """A run where nothing was rejected still produces a readable report."""
    ingest(config, layout, 3, downloader, logger=lambda m: None)
    validate(config, layout, logger=lambda m: None)
    convert(config, layout, logger=lambda m: None)
    split(config, layout, seed=1, logger=lambda m: None)
    data = build(config, layout)
    assert data["counts"]["ingested"] == 3
    assert "none" in render_markdown(data) or "corrupt_image" in render_markdown(data)
