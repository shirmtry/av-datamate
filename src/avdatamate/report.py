"""Stage 5 - quality report (report.json + report.md) for the processed dataset."""

from __future__ import annotations

import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable

from .config import Layout, PipelineConfig, read_jsonl
from .taxonomy import CLASSES

RESOLUTION_BUCKETS = ((320, "tiny"), (640, "sd"), (1280, "hd"), (2560, "fhd"), (10 ** 9, "uhd+"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bucket(longest_side: int) -> str:
    for limit, name in RESOLUTION_BUCKETS:
        if longest_side < limit:
            return name
    return "uhd+"


def build(cfg: PipelineConfig, layout: Layout) -> dict[str, Any]:
    state = layout.read_state()
    validated = read_jsonl(layout.validated)
    rejected = read_jsonl(layout.rejected)
    splits = read_jsonl(layout.splits_index)

    per_class_boxes: Counter[str] = Counter()
    per_class_images: Counter[str] = Counter()
    sizes: list[int] = []
    per_source: Counter[str] = Counter()
    for row in validated:
        per_source[row["source"]] += 1
        sizes.append(max(row["width"], row["height"]))
        seen: set[str] = set()
        for box in row["boxes"]:
            per_class_boxes[box["cls"]] += 1
            seen.add(box["cls"])
        for cls in seen:
            per_class_images[cls] += 1

    split_counts: dict[str, dict[str, int]] = {}
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    class_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in splits:
        part = row["split"]
        entry = split_counts.setdefault(part, {"images": 0, "boxes": 0})
        entry["images"] += 1
        entry["boxes"] += row["n_boxes"]
        groups_by_split[part].add(f"{row['source']}::{row['group']}")
        for cls, n in (row.get("class_counts") or {}).items():
            class_by_split[part][cls] += n

    overlaps = set()
    parts = list(groups_by_split)
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            overlaps |= groups_by_split[a] & groups_by_split[b]

    rejected_reasons = Counter(r["reason"] for r in rejected)
    return {
        "generated_at": _now(),
        "dataset_version": f"v{cfg.version}",
        "pipeline": {
            "seed": (state.get("split") or {}).get("seed"),
            "python": platform.python_version(),
            "config": str(cfg.path),
            "sources_file_version": cfg.version,
        },
        "counts": {
            "ingested": (state.get("ingest") or {}).get("samples", 0),
            "validated": len(validated),
            "rejected": len(rejected),
            "published": sum(v["images"] for v in split_counts.values()),
            "boxes": sum(per_class_boxes.values()),
        },
        "splits": split_counts,
        "split_ratios": (state.get("split") or {}).get("ratios"),
        "groups": {part: len(members) for part, members in groups_by_split.items()},
        "leakage": {"overlapping_groups": len(overlaps), "clean": not overlaps},
        "class_distribution": {
            "boxes": {cls: per_class_boxes.get(cls, 0) for cls in CLASSES},
            "images": {cls: per_class_images.get(cls, 0) for cls in CLASSES},
            "by_split_boxes": {p: {cls: class_by_split[p].get(cls, 0) for cls in CLASSES} for p in split_counts},
            "missing_classes": [cls for cls in CLASSES if per_class_boxes.get(cls, 0) == 0],
        },
        "image_sizes": {
            "n": len(sizes),
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": round(mean(sizes), 1) if sizes else 0,
            "buckets": dict(Counter(_bucket(s) for s in sizes)),
        },
        "per_source": {
            src: {
                "available": (state.get("ingest") or {}).get("available", {}).get(src, 0),
                "accepted": per_source.get(src, 0),
                "license": cfg.source(src).license,
                "pretty_name": cfg.source(src).pretty_name,
            }
            for src in [s.id for s in cfg.sources]
        },
        "rejected_by_reason": dict(rejected_reasons),
        "dropped_boxes_by_reason": (state.get("validate") or {}).get("dropped_boxes", {}),
        "rejections": [
            {"uid": r["uid"], "reason": r["reason"], "detail": r.get("detail", "")}
            for r in rejected
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines: list[str] = [
        f"# av-datamate quality report ({report['dataset_version']})",
        "",
        f"Generated: {report['generated_at']}  ",
        f"Seed: `{report['pipeline']['seed']}`  Python: `{report['pipeline']['python']}`",
        "",
        "## Totals",
        "",
        "| stage | samples |",
        "| --- | --- |",
        f"| ingested | {counts['ingested']} |",
        f"| validated | {counts['validated']} |",
        f"| rejected | {counts['rejected']} |",
        f"| published | {counts['published']} |",
        f"| boxes | {counts['boxes']} |",
        "",
        "## Splits",
        "",
        "| split | images | boxes | scene groups |",
        "| --- | --- | --- | --- |",
    ]
    for part in ("train", "val", "test"):
        entry = report["splits"].get(part, {"images": 0, "boxes": 0})
        lines.append(
            f"| {part} | {entry['images']} | {entry['boxes']} | {report['groups'].get(part, 0)} |"
        )
    leak = report["leakage"]
    leak_note = (
        "PASS - no scene group appears in two splits"
        if leak["clean"]
        else f"FAIL - {leak['overlapping_groups']} groups shared between splits"
    )
    lines += [
        "",
        f"Leakage check: {leak_note}",
        "",
        "## Class distribution",
        "",
        "| class | boxes | images | train | val | test |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    by_split = report["class_distribution"]["by_split_boxes"]
    for cls in CLASSES:
        lines.append(
            f"| {cls} | {report['class_distribution']['boxes'][cls]} | "
            f"{report['class_distribution']['images'][cls]} | "
            + " | ".join(str(by_split.get(p, {}).get(cls, 0)) for p in ("train", "val", "test"))
            + " |"
        )
    missing = report["class_distribution"]["missing_classes"]
    if missing:
        lines += ["", f"Classes with no boxes in this run: {', '.join(missing)}."]
    sizes = report["image_sizes"]
    lines += [
        "",
        "## Image sizes (longest side, px)",
        "",
        f"- n={sizes['n']} min={sizes['min']} max={sizes['max']} mean={sizes['mean']}",
        "- buckets: " + ", ".join(f"{k}: {v}" for k, v in sorted(sizes["buckets"].items())),
        "",
        "## Rejected samples",
        "",
        "| reason | count |",
        "| --- | --- |",
    ]
    if report["rejected_by_reason"]:
        for reason, n in sorted(report["rejected_by_reason"].items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {n} |")
    else:
        lines.append("| none | 0 |")
    dropped = report["dropped_boxes_by_reason"]
    lines += ["", "Boxes dropped during validation: " + (json.dumps(dropped) if dropped else "none"), ""]
    lines += ["## Sources", "", "| dataset | license | accepted | available |", "| --- | --- | --- | --- |"]
    for src, info in report["per_source"].items():
        lines.append(
            f"| `{src}` | {info['license']} | {info['accepted']} | {info['available']} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"avdatamate run --max-samples {report['counts']['ingested']} --out ./out",
        "```",
        "",
    ]
    return "\n".join(lines)


def report(cfg: PipelineConfig, layout: Layout, logger: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict[str, Any]:
    data = build(cfg, layout)
    layout.report_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    layout.report_md.write_text(render_markdown(data), encoding="utf-8")
    logger(
        f"report: {data['counts']['published']} published images, "
        f"{data['counts']['boxes']} boxes, {len(data['class_distribution']['missing_classes'])} empty classes"
    )
    return data
