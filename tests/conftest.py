"""Shared fixtures: a fake Hugging Face dataset repo on disk, no network needed."""

from __future__ import annotations

import json
import random
import shutil
import zlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from avdatamate.config import Layout, PipelineConfig


def make_image(path: Path, size=(320, 240), color=(120, 140, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # content is keyed on the file name so that same-size images are never byte-identical
    rng = random.Random(zlib.crc32(path.name.encode("utf-8")))
    img = Image.new("RGB", size, tuple((c + rng.randrange(48)) % 255 for c in color))
    px = img.load()
    for _ in range(40):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        for dx in range(5):
            for dy in range(5):
                px[min(x + dx, size[0] - 1), min(y + dy, size[1] - 1)] = (
                    rng.randrange(256),
                    rng.randrange(256),
                    rng.randrange(256),
                )
    img.save(path, quality=88)
    return path


def write_label(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


REPO_ID = "fake/traffic-mini"
CLIP_A = "MVI_1001"
CLIP_B = "MVI_2002"


def build_fake_repo(root: Path, extra_bad: bool = True) -> Path:
    """A YOLO-style repo: train/images/<file>.jpg + train/labels/<file>.txt (+ sharded dir)."""
    repo = root / REPO_ID
    repo.mkdir(parents=True, exist_ok=True)
    names = ["car", "bus", "van", "motorcycle", "fence"]
    (repo / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "train/images", "nc": len(names), "names": names}),
        encoding="utf-8",
    )
    good = [
        (f"{CLIP_A}_img00001", ["0 0.5 0.5 0.2 0.3", "1 0.2 0.8 0.1 0.15"]),
        (f"{CLIP_A}_img00002", ["0 0.4 0.5 0.25 0.2"]),
        (f"{CLIP_A}_img00003", ["2 0.5 0.5 0.3 0.3"]),
        (f"{CLIP_B}_img00001", ["0 0.1 0.1 0.2 0.2", "4 0.9 0.9 0.1 0.1"]),
        (f"{CLIP_B}_img00002", ["3 0.5 0.5 0.1 0.2"]),
    ]
    for index, (stem, rows) in enumerate(good):
        sub = "0" if index < 3 else "1"
        make_image(repo / "train" / "images" / sub / f"{stem}.jpg")
        write_label(repo / "train" / "labels" / sub / f"{stem}.txt", rows)
    # a video-only file that must never be picked
    (repo / "videos").mkdir(parents=True, exist_ok=True)
    (repo / "videos" / f"{CLIP_A}.mp4").write_text("not-an-image", encoding="utf-8")
    # image with no matching label
    make_image(repo / "train" / "images" / "2" / f"{CLIP_A}_img00999.jpg")
    if extra_bad:
        # not a decodable jpeg -> corrupt_image
        bad = repo / "train" / "images" / "2" / f"{CLIP_B}_img00998.jpg"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"definitely-not-a-jpeg")
        write_label(repo / "train" / "labels" / "2" / f"{CLIP_B}_img00998.txt", ["0 0.5 0.5 0.2 0.2"])
        # duplicate of img00001 -> duplicate_image
        dup = repo / "train" / "images" / "2" / f"{CLIP_B}_img00997.jpg"
        shutil.copyfile(repo / "train" / "images" / "0" / f"{CLIP_A}_img00001.jpg", dup)
        write_label(repo / "train" / "labels" / "2" / f"{CLIP_B}_img00997.txt", ["0 0.5 0.5 0.2 0.3"])
        # tiny image -> image_too_small
        make_image(repo / "train" / "images" / "2" / f"{CLIP_B}_img00996.jpg", size=(16, 12))
        write_label(repo / "train" / "labels" / "2" / f"{CLIP_B}_img00996.txt", ["0 0.5 0.5 0.2 0.2"])
        # label with every box unmapped -> no_valid_boxes
        make_image(repo / "train" / "images" / "2" / f"{CLIP_B}_img00995.jpg")
        write_label(repo / "train" / "labels" / "2" / f"{CLIP_B}_img00995.txt", ["4 0.5 0.5 0.2 0.2"])
        # empty label -> empty_label
        make_image(repo / "train" / "images" / "2" / f"{CLIP_B}_img00994.jpg")
        write_label(repo / "train" / "labels" / "2" / f"{CLIP_B}_img00994.txt", [])
        # keeps one valid box but exercises every box-level rejection reason
        make_image(repo / "train" / "images" / "2" / f"{CLIP_B}_img00993.jpg")
        write_label(
            repo / "train" / "labels" / "2" / f"{CLIP_B}_img00993.txt",
            [
                "0 0.5 0.5 0.2 0.2",
                "0 1.4 0.5 0.2 0.3",
                "0 0.5 0.5 0.000001 0.000001",
                "44 0.5 0.5 0.1 0.1",
                "4 0.5 0.5 0.2 0.2",
                "garbage row",
            ],
        )
    return repo


def add_clip(repo: Path, clip: str, n: int = 2, start: int = 0) -> list[str]:
    """Append a synthetic clip (n image/label pairs) and return the created stems."""
    stems = []
    for i in range(start, start + n):
        stem = f"{clip}_img{i:05d}"
        make_image(repo / "train" / "images" / "9" / f"{stem}.jpg")
        write_label(repo / "train" / "labels" / "9" / f"{stem}.txt", [f"0 0.{30 + i} 0.5 0.2 0.2"])
        stems.append(stem)
    return stems


class FakeDownloader:
    """Duck-typed stand-in for HfDownloader reading the fixture repo from disk."""

    def __init__(self, root: Path, fail_first: bool = False):
        self.root = root
        self.calls: list[str] = []
        self.listings: int = 0
        self._fail = fail_first

    def listing(self, repo_id: str, repo_type: str = "dataset") -> list[dict[str, Any]]:
        self.listings += 1
        base = self.root / repo_id
        files = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append(
                    {"path": path.relative_to(base).as_posix(), "size": path.stat().st_size}
                )
        return files

    def read_text(self, repo_id: str, path: str, repo_type: str = "dataset") -> str:
        return (self.root / repo_id / path).read_text(encoding="utf-8")

    def fetch(self, repo_id: str, path: str, repo_type: str = "dataset") -> Path:
        if self._fail:
            self._fail = False
            raise OSError("simulated transient network error")
        self.calls.append(path)
        return self.root / repo_id / path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    return build_fake_repo(tmp_path)


@pytest.fixture
def downloader(fake_repo: Path) -> FakeDownloader:
    # FakeDownloader is rooted at the parent of <repo_id>, like a HF cache root.
    return FakeDownloader(fake_repo.parent.parent)


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    text = {
        "version": 1,
        "target_classes": ["car", "truck", "bus", "pedestrian", "cyclist", "traffic_light", "traffic_sign"],
        "sources": [
            {
                "id": REPO_ID,
                "repo_type": "dataset",
                "license": "mit",
                "pretty_name": "Fake traffic mini",
                "format": "yolo_dir",
                "group_regex": r"^(MVI_\d+)",
                "exclude_prefixes": ["videos/"],
                "class_map": {"van": "car", "motorcycle": "cyclist"},
            }
        ],
    }
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(text), encoding="utf-8")
    return PipelineConfig.load(path)


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    return Layout(tmp_path / "out")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
