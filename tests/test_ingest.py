from pathlib import Path

from conftest import REPO_ID, FakeDownloader, load_jsonl

from avdatamate.config import Layout, PipelineConfig
from avdatamate.ingest import allocate_quotas, group_key, ingest, label_path_for, uid_for


def test_label_path_for_handles_nested_images():
    assert label_path_for("train/images/0/a.jpg") == "train/labels/0/a.txt"
    assert label_path_for("images/train/clip#t=1.jpg") == "labels/train/clip#t=1.txt"
    assert label_path_for("photos/a.jpg") == ""


def test_uid_and_group_are_stable(config: PipelineConfig):
    source = config.source(REPO_ID)
    assert uid_for(source, "train/images/0/MVI_1001_img00001.jpg").startswith("fake_traffic-mini__")
    assert group_key(source, "train/images/0/MVI_1001_img00001.jpg") == "MVI_1001"
    assert group_key(source, "train/images/0/noprefix.jpg") == "noprefix"


def test_allocate_quotas_redistributes_surplus():
    avail = {"a": 2, "b": 100, "c": 100}
    quotas = allocate_quotas(avail, 12)
    assert sum(quotas.values()) == 12
    assert quotas["a"] == 2


def test_ingest_downloads_pairs_and_skips_video(config: PipelineConfig, layout: Layout, downloader):
    result = ingest(config, layout, max_samples=50, downloader=downloader, logger=lambda m: None)
    rows = load_jsonl(layout.manifest)
    assert result.selected == len(rows) == 11
    assert all(Path(layout.abs(r["image"])).exists() for r in rows)
    assert all(Path(layout.abs(r["label"])).exists() for r in rows)
    assert not any("videos/" in r["repo_image"] for r in rows)
    assert rows[0]["class_names"] == ["car", "bus", "van", "motorcycle", "fence"]
    state = layout.read_state()["ingest"]
    assert state["available"][REPO_ID] == 11


def test_ingest_respects_max_samples(config: PipelineConfig, layout: Layout, downloader):
    result = ingest(config, layout, max_samples=3, downloader=downloader, logger=lambda m: None)
    assert result.selected == 3
    assert len(load_jsonl(layout.manifest)) == 3


def test_ingest_resumes_without_redownloading(config: PipelineConfig, layout: Layout, downloader):
    ingest(config, layout, 5, downloader, logger=lambda m: None)
    first_calls = len(downloader.calls)
    again = FakeDownloader(downloader.root)
    again.calls = []
    result = ingest(config, layout, 5, again, logger=lambda m: None)
    assert len(load_jsonl(layout.manifest)) == 5
    assert again.calls == [] or result.skipped_existing == 5
    assert first_calls > 0
