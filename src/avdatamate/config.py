"""Loading of configs/sources.yaml plus the shared output-directory layout."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .taxonomy import CLASSES

MIN_PYTHON = (3, 10)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class SourceConfig:
    id: str
    repo_type: str = "dataset"
    license: str = ""
    license_source: str = ""
    upstream_samples: int = 0
    pretty_name: str = ""
    upstream_repo: str = ""
    format: str = "yolo_dir"
    group_regex: str = ""
    class_map: dict[str, str] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    exclude_prefixes: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", self.id)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SourceConfig:
        allowed = {f for f in SourceConfig.__dataclass_fields__}
        kwargs = {k: v for k, v in raw.items() if k in allowed}
        kwargs["class_map"] = {str(k): str(v) for k, v in (raw.get("class_map") or {}).items()}
        kwargs["class_names"] = [str(c) for c in (raw.get("class_names") or [])]
        kwargs["exclude_prefixes"] = [str(p) for p in (raw.get("exclude_prefixes") or [])]
        if "id" not in kwargs:
            raise ConfigError(f"source without id: {raw!r}")
        return SourceConfig(**kwargs)


@dataclass
class PipelineConfig:
    version: int
    target_classes: list[str]
    sources: list[SourceConfig]
    path: Path | None = None

    def source(self, repo_id: str) -> SourceConfig:
        for src in self.sources:
            if src.id == repo_id:
                return src
        raise ConfigError(f"unknown source {repo_id}")

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        targets = [str(c) for c in raw.get("target_classes") or CLASSES]
        if targets != list(CLASSES):
            raise ConfigError(
                f"target_classes must be {list(CLASSES)}, got {targets}"
            )
        sources = [SourceConfig.from_dict(s) for s in raw.get("sources") or []]
        if not sources:
            raise ConfigError("no sources defined")
        return cls(
            version=int(raw.get("version") or 1),
            target_classes=targets,
            sources=sources,
            path=Path(path),
        )


class ConfigError(RuntimeError):
    pass


class Layout:
    """Filesystem contract between pipeline stages (paths relative to out_dir)."""

    def __init__(self, out_dir: Path | str):
        self.root = Path(out_dir)
        self.raw = self.root / "raw"
        self.raw_images = self.raw / "images"
        self.raw_labels = self.raw / "labels"
        self.manifest = self.raw / "manifest.jsonl"
        self.validated = self.root / "validated.jsonl"
        self.rejected = self.root / "rejected.jsonl"
        self.converted = self.root / "converted"
        self.converted_images = self.converted / "images"
        self.converted_labels = self.converted / "labels"
        self.converted_data_yaml = self.converted / "data.yaml"
        self.converted_index = self.converted / "index.jsonl"
        self.dataset = self.root / "dataset"
        self.dataset_images = self.dataset / "images"
        self.dataset_labels = self.dataset / "labels"
        self.dataset_data_yaml = self.dataset / "data.yaml"
        self.splits_index = self.dataset / "splits.jsonl"
        self.report_json = self.root / "report.json"
        self.report_md = self.root / "report.md"
        self.state = self.root / "state.json"

    def ensure_dirs(self) -> None:
        for d in (self.raw_images, self.raw_labels):
            d.mkdir(parents=True, exist_ok=True)

    def rel(self, path: Path | str) -> str:
        return Path(path).relative_to(self.root).as_posix()

    def abs(self, rel: str | Path) -> Path:
        return self.root / rel

    def write_state(self, key: str, value: Any) -> None:
        data = {}
        if self.state.exists():
            try:
                data = json.loads(self.state.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data[key] = value
        self.state.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def read_state(self) -> dict[str, Any]:
        if not self.state.exists():
            return {}
        return json.loads(self.state.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(fh, row: dict[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(
            f"python >= {'.'.join(map(str, MIN_PYTHON))} required, found "
            f"{'.'.join(map(str, sys.version_info[:3]))}"
        )
