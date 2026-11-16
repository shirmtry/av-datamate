"""Stage 3 - emit the validated pool as a YOLO-format dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

from .config import Layout, PipelineConfig, append_jsonl, read_jsonl
from .taxonomy import CLASS_TO_IDX, CLASSES

EXT_BY_FORMAT = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp"}


@dataclass
class ConvertResult:
    images: int = 0
    boxes: int = 0
    per_class: dict[str, int] = field(default_factory=dict)


def yolo_label_text(boxes: list[dict[str, Any]]) -> str:
    lines = []
    for box in boxes:
        idx = CLASS_TO_IDX[box["cls"]]
        lines.append(
            f"{idx} {_r6(box['cx'])} {_r6(box['cy'])} {_r6(box['w'])} {_r6(box['h'])}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _r6(value: float) -> str:
    return f"{round(float(value), 6):.6f}"


def data_yaml(image_dirs: dict[str, str]) -> str:
    """`path: .` keeps the yaml portable: YOLO resolves it relative to the file itself."""
    doc = {
        "path": ".",
        **image_dirs,
        "nc": len(CLASSES),
        "names": list(CLASSES),
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def convert(
    cfg: PipelineConfig,
    layout: Layout,
    logger: Callable[[str], None] = lambda m: print(m, flush=True),
) -> ConvertResult:
    rows = read_jsonl(layout.validated)
    layout.converted_images.mkdir(parents=True, exist_ok=True)
    layout.converted_labels.mkdir(parents=True, exist_ok=True)
    result = ConvertResult()
    index_path = layout.converted_index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as idx:
        for row in rows:
            src_image = layout.abs(row["image"])
            ext = EXT_BY_FORMAT.get(str(row.get("format", "")).lower(), src_image.suffix.lower() or ".jpg")
            image_rel = f"converted/images/{row['uid']}{ext}"
            label_rel = f"converted/labels/{row['uid']}.txt"
            shutil.copyfile(src_image, layout.abs(image_rel))
            layout.abs(label_rel).write_text(yolo_label_text(row["boxes"]), encoding="utf-8")
            counts = {c: sum(1 for b in row["boxes"] if b["cls"] == c) for c in {b["cls"] for b in row["boxes"]}}
            append_jsonl(
                idx,
                {
                    "uid": row["uid"],
                    "source": row["source"],
                    "group": row["group"],
                    "image": image_rel,
                    "label": label_rel,
                    "width": row["width"],
                    "height": row["height"],
                    "sha256": row["sha256"],
                    "n_boxes": len(row["boxes"]),
                    "classes": sorted(counts),
                    "class_counts": counts,
                },
            )
            result.images += 1
            result.boxes += len(row["boxes"])
            for cls, n in counts.items():
                result.per_class[cls] = result.per_class.get(cls, 0) + n
    layout.converted_data_yaml.write_text(
        data_yaml({"train": "images", "val": "images", "test": "images"}),
        encoding="utf-8",
    )
    layout.write_state(
        "convert",
        {
            "images": result.images,
            "boxes": result.boxes,
            "per_class": result.per_class,
            "classes": list(CLASSES),
        },
    )
    logger(f"convert: {result.images} images / {result.boxes} boxes -> {layout.converted}")
    return result
