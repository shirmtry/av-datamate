"""Stage 2 - drop broken, duplicated and geometrically invalid annotation samples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image, UnidentifiedImageError

from .config import Layout, PipelineConfig, SourceConfig, append_jsonl, read_jsonl
from .taxonomy import resolve_class

MIN_SIDE_PX = 32
MIN_REL_SIDE = 0.002
MIN_REL_AREA = 1e-6
MIN_AREA_PX = 25.0
BOUNDS_EPS = 1e-3

REJECT_REASONS = (
    "corrupt_image",
    "duplicate_image",
    "image_too_small",
    "missing_label",
    "malformed_label",
    "empty_label",
    "unknown_class_index",
    "unmapped_class",
    "bbox_out_of_bounds",
    "bbox_degenerate",
    "no_valid_boxes",
)


@dataclass
class BoxCheck:
    cls: str
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo(self) -> str:
        from .taxonomy import CLASS_TO_IDX

        return f"{CLASS_TO_IDX[self.cls]} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ValidateStats:
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    dropped_boxes: dict[str, int] = field(default_factory=dict)

    def add_reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def add_box(self, reason: str) -> None:
        self.dropped_boxes[reason] = self.dropped_boxes.get(reason, 0) + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_class(index: int, class_names: list[str], source: SourceConfig) -> str | None:
    if index < 0 or index >= len(class_names):
        return None
    name = class_names[index]
    mapped = source.class_map.get(name)
    if mapped is not None:
        return mapped if resolve_class(mapped) else None
    return resolve_class(name)


def parse_label_file(
    path: Path, class_names: list[str], source: SourceConfig, width: int, height: int,
    stats: ValidateStats,
) -> tuple[list[BoxCheck], list[str], int]:
    boxes: list[BoxCheck] = []
    problems: list[str] = []
    n_rows = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        problems.append(f"missing_label:{exc}")
        stats.add_box("missing_label")
        return boxes, problems, n_rows
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        n_rows += 1
        parts = line.split()
        if len(parts) < 5:
            problems.append(f"malformed_label:{line[:40]}")
            stats.add_box("malformed_label")
            continue
        try:
            index = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            problems.append(f"malformed_label:{line[:40]}")
            stats.add_box("malformed_label")
            continue
        if not all(_finite(v) for v in (cx, cy, w, h)):
            stats.add_box("malformed_label")
            continue
        cls = canonical_class(index, class_names, source)
        if cls is None:
            stats.add_box(
                "unknown_class_index" if index < 0 or index >= len(class_names) else "unmapped_class"
            )
            continue
        if cx < -BOUNDS_EPS or cx > 1 + BOUNDS_EPS or cy < -BOUNDS_EPS or cy > 1 + BOUNDS_EPS:
            stats.add_box("bbox_out_of_bounds")
            continue
        if w <= 0 or h <= 0:
            stats.add_box("bbox_degenerate")
            continue
        if cx - w / 2 < -BOUNDS_EPS or cx + w / 2 > 1 + BOUNDS_EPS:
            stats.add_box("bbox_out_of_bounds")
            continue
        if cy - h / 2 < -BOUNDS_EPS or cy + h / 2 > 1 + BOUNDS_EPS:
            stats.add_box("bbox_out_of_bounds")
            continue
        if min(w, h) < MIN_REL_SIDE or w * h < MIN_REL_AREA:
            stats.add_box("bbox_degenerate")
            continue
        if w * width * h * height < MIN_AREA_PX:
            stats.add_box("bbox_degenerate")
            continue
        boxes.append(
            BoxCheck(
                cls=cls,
                cx=min(max(cx, 0.0), 1.0),
                cy=min(max(cy, 0.0), 1.0),
                w=min(w, 1.0),
                h=min(h, 1.0),
            )
        )
    return boxes, problems, n_rows


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def iter_samples(manifest: Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(manifest)


def validate(
    cfg: PipelineConfig,
    layout: Layout,
    logger: Callable[[str], None] = lambda m: print(m, flush=True),
) -> ValidateStats:
    rows = read_jsonl(layout.manifest)
    stats = ValidateStats()
    by_source = {s.id: s for s in cfg.sources}
    seen_hashes: dict[str, str] = {}
    with layout.validated.open("w", encoding="utf-8") as out, layout.rejected.open(
        "w", encoding="utf-8"
    ) as bad:
        for row in rows:
            source = by_source.get(row["source"])
            image_path = layout.abs(row["image"])
            if source is None:
                _reject(bad, row, "unmapped_class", "source not in config")
                stats.add_reject("unmapped_class")
                continue
            if not image_path.exists():
                _reject(bad, row, "corrupt_image", "file missing on disk")
                stats.add_reject("corrupt_image")
                continue
            try:
                digest = sha256_file(image_path)
            except OSError as exc:
                _reject(bad, row, "corrupt_image", str(exc))
                stats.add_reject("corrupt_image")
                continue
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    width, height = im.size
                    fmt = (im.format or Path(image_path).suffix.lstrip(".")).lower()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                _reject(bad, row, "corrupt_image", str(exc))
                stats.add_reject("corrupt_image")
                continue
            if digest in seen_hashes:
                _reject(bad, row, "duplicate_image", f"same as {seen_hashes[digest]}")
                stats.add_reject("duplicate_image")
                continue
            if min(width, height) < MIN_SIDE_PX:
                _reject(bad, row, "image_too_small", f"{width}x{height}")
                stats.add_reject("image_too_small")
                continue
            label_path = layout.abs(row["label"])
            if not label_path.exists():
                _reject(bad, row, "missing_label", str(row["label"]))
                stats.add_reject("missing_label")
                continue
            class_names = row.get("class_names") or source.class_names
            boxes, problems, n_rows = parse_label_file(
                label_path, class_names, source, width, height, stats
            )
            if not boxes:
                reason = "empty_label" if n_rows == 0 else "no_valid_boxes"
                _reject(bad, row, reason, ";".join(problems[:3]) or "no usable boxes in label file")
                stats.add_reject(reason)
                seen_hashes[digest] = row["uid"]
                continue
            seen_hashes[digest] = row["uid"]
            append_jsonl(
                out,
                {
                    "uid": row["uid"],
                    "source": row["source"],
                    "group": row["group"],
                    "image": row["image"],
                    "format": fmt,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                    "boxes": [json.loads(json.dumps(b.__dict__)) for b in boxes],
                    "repo_image": row["repo_image"],
                    "label_problems": problems[:5],
                },
            )
            stats.accepted += 1
    layout.write_state(
        "validate",
        {
            "accepted": stats.accepted,
            "rejected": stats.rejected,
            "dropped_boxes": stats.dropped_boxes,
            "candidates": len(rows),
        },
    )
    logger(
        f"validate: kept {stats.accepted}/{len(rows)} samples, "
        f"rejected={json.dumps(stats.rejected)}, dropped_boxes={json.dumps(stats.dropped_boxes)}"
    )
    return stats


def _reject(fh, row: dict[str, Any], reason: str, detail: str) -> None:
    append_jsonl(
        fh,
        {
            "uid": row.get("uid"),
            "source": row.get("source"),
            "group": row.get("group"),
            "image": row.get("image"),
            "reason": reason,
            "detail": detail,
        },
    )
