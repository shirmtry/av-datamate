"""Stage 4 - leak-free train/val/test split with a fixed seed."""

from __future__ import annotations

import hashlib
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Layout, PipelineConfig, append_jsonl, read_jsonl
from .convert import data_yaml

RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass
class SplitResult:
    counts: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, int] = field(default_factory=dict)
    groups: int = 0


def group_id(row: dict[str, Any]) -> str:
    return f"{row['source']}::{row['group']}"


def _stable_rank(key: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode()).digest()


def assign_groups(groups: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, str]:
    """Give whole groups to one split each so frames of a clip never straddle splits."""
    total = sum(len(v) for v in groups.values())
    keys = sorted(groups, key=lambda k: (_stable_rank(k, seed), k))
    targets = {split: total * ratio for split, ratio in RATIOS.items()}
    filled = {split: 0 for split in RATIOS}
    assignment: dict[str, str] = {}
    for key in keys:
        # most under-filled split, ties broken by the canonical order
        split = min(RATIOS, key=lambda s: (filled[s] - targets[s], list(RATIOS).index(s)))
        assignment[key] = split
        filled[split] += len(groups[key])
    _ensure_coverage(assignment, groups, filled)
    return assignment


def _ensure_coverage(
    assignment: dict[str, str], groups: dict[str, list[dict[str, Any]]], filled: dict[str, int]
) -> None:
    """Whole clips are coarse: move the smallest donor group so val/test are never empty."""
    for split in ("val", "test"):
        if filled[split] or len(groups) < 3:
            continue
        donor_split = max(filled, key=lambda s: (filled[s], -list(RATIOS).index(s)))
        donors = [k for k, s in assignment.items() if s == donor_split]
        if len(donors) < 2:
            continue
        donor = min(donors, key=lambda k: (len(groups[k]), k))
        moved = len(groups[donor])
        assignment[donor] = split
        filled[donor_split] -= moved
        filled[split] += moved


def split(
    cfg: PipelineConfig,
    layout: Layout,
    seed: int = 1337,
    logger: Callable[[str], None] = lambda m: print(m, flush=True),
) -> SplitResult:
    rows = read_jsonl(layout.converted_index)
    if not rows:
        raise RuntimeError("converted index is empty - run convert first")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_id(row)].append(row)
    assignment = assign_groups(groups, seed)

    layout.dataset_images.mkdir(parents=True, exist_ok=True)
    layout.dataset_labels.mkdir(parents=True, exist_ok=True)
    _clear_dataset(layout)
    for part in RATIOS:
        (layout.dataset_images / part).mkdir(parents=True, exist_ok=True)
        (layout.dataset_labels / part).mkdir(parents=True, exist_ok=True)

    counts = {part: 0 for part in RATIOS}
    boxes = {part: 0 for part in RATIOS}
    with layout.splits_index.open("w", encoding="utf-8") as idx:
        for row in rows:
            part = assignment[group_id(row)]
            src_image = layout.abs(row["image"])
            dst_image_rel = f"dataset/images/{part}/{row['uid']}{src_image.suffix}"
            dst_label_rel = f"dataset/labels/{part}/{row['uid']}.txt"
            shutil.copyfile(src_image, layout.abs(dst_image_rel))
            shutil.copyfile(layout.abs(row["label"]), layout.abs(dst_label_rel))
            record = dict(row)
            record["split"] = part
            record["image"] = dst_image_rel
            record["label"] = dst_label_rel
            append_jsonl(idx, record)
            counts[part] += 1
            boxes[part] += row["n_boxes"]

    leaked = _check_leakage(rows, assignment)
    if leaked:
        raise RuntimeError(f"split leakage detected: {sorted(leaked)[:5]}")
    layout.dataset_data_yaml.write_text(
        data_yaml(
            {"train": "images/train", "val": "images/val", "test": "images/test"}
        ),
        encoding="utf-8",
    )
    layout.write_state(
        "split",
        {"seed": seed, "counts": counts, "boxes": boxes, "groups": len(groups), "ratios": RATIOS},
    )
    logger(f"split(seed={seed}): {counts} over {len(groups)} scene groups")
    return SplitResult(counts=counts, boxes=boxes, groups=len(groups))


def _clear_dataset(layout: Layout) -> None:
    """Drop images/labels left over from an earlier run so the dataset matches splits.jsonl."""
    for base in (layout.dataset_images, layout.dataset_labels):
        for path in base.rglob("*"):
            if path.is_file():
                path.unlink()


def _check_leakage(rows: list[dict[str, Any]], assignment: dict[str, str]) -> set[str]:
    seen: dict[str, str] = {}
    bad: set[str] = set()
    for row in rows:
        key = group_id(row)
        part = assignment[key]
        if key in seen and seen[key] != part:
            bad.add(key)
        seen[key] = part
    return bad
