from collections import Counter

from conftest import load_jsonl

from avdatamate.config import Layout, PipelineConfig
from avdatamate.ingest import ingest
from avdatamate.validate import validate


def run_ingest_and_validate(config: PipelineConfig, layout: Layout, downloader):
    ingest(config, layout, 100, downloader, logger=lambda m: None)
    stats = validate(config, layout, logger=lambda m: None)
    return stats


def test_validate_accepts_good_and_rejects_bad(config, layout, downloader):
    stats = run_ingest_and_validate(config, layout, downloader)
    kept = load_jsonl(layout.validated)
    rejected = load_jsonl(layout.rejected)
    reasons = Counter(r["reason"] for r in rejected)
    assert stats.accepted == len(kept) == 6
    assert reasons == Counter(
        {"corrupt_image": 1, "duplicate_image": 1, "image_too_small": 1, "empty_label": 1, "no_valid_boxes": 1}
    )
    assert layout.read_state()["validate"]["candidates"] == 11


def test_validate_maps_classes_through_class_map(config, layout, downloader):
    run_ingest_and_validate(config, layout, downloader)
    kept = {row["uid"]: row for row in load_jsonl(layout.validated)}
    row = next(r for r in kept.values() if r["repo_image"].endswith("MVI_1001_img00001.jpg"))
    assert [b["cls"] for b in row["boxes"]] == ["car", "bus"]
    moto = next(r for r in kept.values() if r["repo_image"].endswith("MVI_2002_img00002.jpg"))
    assert [b["cls"] for b in moto["boxes"]] == ["cyclist"]


def test_validate_drops_each_kind_of_bad_box(config, layout, downloader):
    run_ingest_and_validate(config, layout, downloader)
    dropped = layout.read_state()["validate"]["dropped_boxes"]
    assert dropped["bbox_out_of_bounds"] == 1
    assert dropped["bbox_degenerate"] == 1
    assert dropped["unknown_class_index"] == 1
    assert dropped["unmapped_class"] == 3
    assert dropped["malformed_label"] == 1
    kept = {r["repo_image"].rsplit("/", 1)[1]: r for r in load_jsonl(layout.validated)}
    assert len(kept["MVI_2002_img00993.jpg"]["boxes"]) == 1


def test_validate_records_geometry_and_hash(config, layout, downloader):
    run_ingest_and_validate(config, layout, downloader)
    kept = load_jsonl(layout.validated)
    assert all(len(r["sha256"]) == 64 for r in kept)
    assert all(r["width"] == 320 and r["height"] == 240 for r in kept)
    for row in kept:
        for box in row["boxes"]:
            assert 0.0 <= box["cx"] <= 1.0 and 0.0 < box["w"] <= 1.0


def test_canonical_class_prefers_explicit_map():
    from avdatamate.config import SourceConfig
    from avdatamate.validate import canonical_class

    source = SourceConfig(id="a/b", class_map={"van": "car"})
    names = ["car", "van", "person", "traffic sign", "nothing"]
    assert canonical_class(0, names, source) == "car"
    assert canonical_class(1, names, source) == "car"
    assert canonical_class(2, names, source) == "pedestrian"
    assert canonical_class(3, names, source) == "traffic_sign"
    assert canonical_class(4, names, source) is None
    assert canonical_class(9, names, source) is None
