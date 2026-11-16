import json
from pathlib import Path

import pytest
from conftest import FakeDownloader, build_fake_repo

from avdatamate import cli


@pytest.fixture
def fake_downloader(tmp_path: Path):
    build_fake_repo(tmp_path / "hub")
    return FakeDownloader(tmp_path / "hub")


def patch_cli(monkeypatch, downloader):
    monkeypatch.setattr(cli, "HfDownloader", lambda *a, **k: downloader)


def test_run_produces_report_and_yolo_dataset(tmp_path, config, monkeypatch, fake_downloader):
    patch_cli(monkeypatch, fake_downloader)
    out = tmp_path / "pipeline-out"
    code = cli.main(["--config", str(config.path), "--out", str(out), "--quiet", "run", "--max-samples", "8"])
    assert code == 0
    assert (out / "raw" / "manifest.jsonl").exists()
    assert (out / "dataset" / "data.yaml").exists()
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    assert list((out / "dataset" / "images" / "train").glob("*.jpg"))
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["ingest"]["samples"] == 8
    assert state["split"]["seed"] == 1337
    # raw downloads are not part of the published payload
    assert not (out / "dataset" / "raw").exists()


def test_stage_commands_run_individually(tmp_path, config, monkeypatch, fake_downloader):
    downloader = fake_downloader
    patch_cli(monkeypatch, downloader)
    out = tmp_path / "stages"
    argv = ["--config", str(config.path), "--out", str(out)]
    assert cli.main(argv + ["ingest", "--max-samples", "4"]) == 0
    assert len(downloader.calls) > 0
    n_after_ingest = len(downloader.calls)
    assert cli.main(argv + ["validate"]) == 0
    assert cli.main(argv + ["convert"]) == 0
    assert cli.main(argv + ["split", "--seed", "5"]) == 0
    assert cli.main(argv + ["report"]) == 0
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert state["split"]["seed"] == 5
    # a second ingest resumes instead of re-downloading
    assert cli.main(argv + ["ingest", "--max-samples", "4"]) == 0
    assert len(downloader.calls) == n_after_ingest


def test_publish_dry_run_does_not_touch_the_hub(tmp_path, config, monkeypatch, fake_downloader):
    patch_cli(monkeypatch, fake_downloader)
    out = tmp_path / "pub"
    assert cli.main(["--config", str(config.path), "--out", str(out), "--quiet", "run", "--max-samples", "6"]) == 0
    import avdatamate.publish as publish

    monkeypatch.setenv("HF_TOKEN", "x" * 8)

    class FakeApi:
        def __init__(self):
            self.uploads = []

        def repo_info(self, repo_id, repo_type=None):
            if repo_type != "model":
                raise RuntimeError("404")
            return object()

        def upload_folder(self, **kwargs):
            self.uploads.append(kwargs)

    fake = FakeApi()
    monkeypatch.setattr(publish, "_api", lambda token: fake)
    assert cli.main(["--config", str(config.path), "--out", str(out), "--quiet", "publish"]) == 0
    assert len(fake.uploads) == 1
    call = fake.uploads[0]
    assert call["repo_type"] == "model"
    assert call["path_in_repo"] == "av-datamate/v1"
    assert call["allow_patterns"]
    assert Path(out / "publish.json").exists()
