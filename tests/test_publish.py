import importlib.util
import os
from pathlib import Path

import pytest
import yaml
from conftest import load_jsonl

from avdatamate.config import Layout, PipelineConfig
from avdatamate.convert import convert
from avdatamate.ingest import ingest
from avdatamate.publish import (
    PublishError,
    _collect_payload,
    detect_repo_type,
    load_token_file,
    publish_dataset,
)
from avdatamate.validate import validate


@pytest.fixture
def published_layout(config: PipelineConfig, layout: Layout, downloader) -> Layout:
    ingest(config, layout, 100, downloader, logger=lambda m: None)
    validate(config, layout, logger=lambda m: None)
    convert(config, layout, logger=lambda m: None)
    from avdatamate.split import split

    split(config, layout, seed=2, logger=lambda m: None)
    from avdatamate.report import report

    report(config, layout, logger=lambda m: None)
    return layout


def test_collect_payload_excludes_raw(published_layout: Layout):
    payload = _collect_payload(published_layout)
    assert any(p.startswith("dataset/images/") for p in payload)
    assert any(p.startswith("dataset/labels/") for p in payload)
    assert "report.json" in payload
    assert "report.md" in payload
    assert not any(p.startswith("raw/") for p in payload)
    assert not any("cache" in p for p in payload)


def test_publish_requires_dataset(config, layout):
    with pytest.raises(PublishError):
        publish_dataset(config, layout, "x/y", "v1", token="t")


def test_load_token_file_strips_bom_and_whitespace(tmp_path, monkeypatch):
    secret = tmp_path / "HF.txt"
    secret.write_bytes("﻿dummy_token_value\r\n\r\n".encode())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_token_file(secret) == "dummy_token_value"
    assert os.environ["HF_TOKEN"] == "dummy_token_value"
    with pytest.raises(PublishError):
        load_token_file(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(PublishError):
        load_token_file(empty)


def test_detect_repo_type_tries_each_kind():
    class Api:
        def __init__(self):
            self.seen = []

        def repo_info(self, repo_id, repo_type=None):
            self.seen.append(repo_type)
            if repo_type != "dataset":
                raise RuntimeError("404")
            return object()

    api = Api()
    assert detect_repo_type(api, "a/b") == "dataset"
    assert api.seen == ["model", "dataset"]


def test_detect_repo_type_raises_when_absent():
    class Api:
        def repo_info(self, repo_id, repo_type=None):
            raise RuntimeError("404")

    with pytest.raises(PublishError):
        detect_repo_type(Api(), "a/b")


def test_publish_dataset_upload_targets_version_folder(published_layout, config, monkeypatch):
    calls = {}

    class Api:
        def repo_info(self, repo_id, repo_type=None):
            if repo_type != "model":
                raise RuntimeError("404")
            return object()

        def upload_folder(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("avdatamate.publish._api", lambda token: Api())
    result = publish_dataset(config, published_layout, "Huy1223/datamate_models", "v9", token="tok")
    assert result["repo_type"] == "model"
    assert result["path_in_repo"] == "av-datamate/v9"
    assert calls["repo_id"] == "Huy1223/datamate_models"
    assert calls["commit_message"].startswith("av-datamate v9")
    assert set(calls["allow_patterns"]) == set(_collect_payload(published_layout))
    assert json_(published_layout.root / "publish.json")["files"] == result["files"]


def json_(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_space_readme_has_valid_front_matter():
    spec = importlib.util.spec_from_file_location(
        "publish_hf", Path(__file__).resolve().parents[1] / "scripts" / "publish_hf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.space_readme("av-datamate viewer", "short text", "mit")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---")[1])
    assert front["sdk"] == "docker"
    assert front["app_port"] == module.APP_PORT
    assert front["title"] == "av-datamate viewer"
    assert front["short_description"] == "short text"
    assert "app_file: app/app.py" in text
    # the generated file must never carry a token
    assert "hf_" not in text


def test_publish_payload_files_exist(published_layout):
    payload = _collect_payload(published_layout)
    assert payload
    for rel in payload:
        assert (published_layout.root / rel).is_file()
    index = load_jsonl(published_layout.splits_index)
    assert index and all("split" in row for row in index)
