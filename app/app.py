"""Gradio viewer for the processed YOLO dataset: images with bboxes plus quality stats."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from avdatamate.taxonomy import CLASSES  # noqa: E402

PREFIX = os.environ.get("AV_PREFIX", "av-datamate")
VERSION = os.environ.get("AV_VERSION", "v1")
SOURCE_REPO = os.environ.get("AV_SOURCE_REPO", "Huy1223/datamate_models")
LOCAL_OUT = Path(os.environ.get("AV_DATA_DIR") or os.environ.get("OUT_DIR") or (REPO_ROOT / "out"))
CACHE_ROOT = Path(os.environ.get("AV_CACHE_DIR", tempfile.gettempdir())) / "av-datamate-cache"
MAX_GALLERY = int(os.environ.get("AV_MAX_GALLERY", "24"))

CLASS_COLORS = {
    "car": (30, 144, 255),
    "truck": (255, 140, 0),
    "bus": (220, 20, 60),
    "pedestrian": (34, 139, 34),
    "cyclist": (148, 0, 211),
    "traffic_light": (255, 215, 0),
    "traffic_sign": (0, 206, 209),
}


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def hub_dir() -> Path:
    return CACHE_ROOT / SOURCE_REPO.replace("/", "__")


def dataset_root() -> Path:
    """Prefer a local pipeline run, otherwise pull the published dataset from the hub."""
    local = LOCAL_OUT / "dataset"
    if (local / "splits.jsonl").exists() or any((local / "images").glob("*/*")):
        return local
    version_dir = hub_dir() / PREFIX / VERSION
    if not version_dir.is_dir():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=SOURCE_REPO,
            repo_type=_repo_type(),
            allow_patterns=[f"{PREFIX}/{VERSION}/**"],
            local_dir=str(hub_dir()),
            token=_token(),
            max_workers=8,
        )
    return version_dir / "dataset"


def _repo_type() -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    for repo_type in ("model", "dataset", "space"):
        try:
            api.repo_info(SOURCE_REPO, repo_type=repo_type)
            return repo_type
        except Exception:
            continue
    return "model"


def _font():
    for candidate in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, 14)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    """boxes use YOLO normalized cx/cy/w/h with a canonical class name."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    width, height = canvas.size
    for box in boxes:
        cls = box.get("cls", "")
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cx, cy = float(box["cx"]) * width, float(box["cy"]) * height
        bw, bh = float(box["w"]) * width, float(box["h"]) * height
        x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{cls} {int(bw)}x{int(bh)}"
        tw = max(30, len(label) * 8)
        ty = max(0, y0 - 16)
        draw.rectangle([x0, ty, min(x0 + tw, width), ty + 16], fill=color)
        draw.text((x0 + 2, ty + 1), label, fill=(0, 0, 0), font=font)
    return canvas


def label_to_boxes(text: str) -> list[dict[str, Any]]:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        idx = int(float(parts[0]))
        if not 0 <= idx < len(CLASSES):
            continue
        boxes.append(
            {
                "cls": CLASSES[idx],
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
        )
    return boxes


def _to_dataset_relative(rel: str) -> str:
    """splits.jsonl stores paths relative to the pipeline out dir; the viewer works from dataset/."""
    posix = str(rel).replace("\\", "/")
    parts = Path(posix).parts
    if parts and parts[0] == "dataset":
        return "/".join(parts[1:])
    return posix


def load_index(root: Path) -> list[dict[str, Any]]:
    index = root / "splits.jsonl"
    if index.exists():
        rows = []
        for line in index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("image"):
                    row["image"] = _to_dataset_relative(row["image"])
                if row.get("label"):
                    row["label"] = _to_dataset_relative(row["label"])
                rows.append(row)
        if rows:
            return rows
    rows = []
    for split in ("train", "val", "test"):
        for image in sorted((root / "images" / split).glob("*")):
            rows.append(
                {
                    "split": split,
                    "source": "",
                    "image": f"images/{split}/{image.name}",
                    "label": f"labels/{split}/{image.stem}.txt",
                }
            )
    return rows


def row_boxes(root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Boxes come inline in the validated index, otherwise read the YOLO label file."""
    boxes = row.get("boxes")
    if boxes:
        return boxes
    label_rel = row.get("label")
    if not label_rel:
        return []
    label = root / label_rel
    if not label.exists():
        return []
    return label_to_boxes(label.read_text(encoding="utf-8"))


def sample_gallery(root: Path, split: str, cls: str, limit: int) -> list[tuple[Image.Image, str]]:
    out: list[tuple[Image.Image, str]] = []
    for row in load_index(root):
        if row.get("split") != split:
            continue
        boxes = row_boxes(root, row)
        if cls != "all" and not any(b["cls"] == cls for b in boxes):
            continue
        path = root / row["image"]
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                out.append((annotate(im, boxes), f"{row.get('source', '')} | {len(boxes)} boxes"))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def stats_table(root: Path) -> list[list[str]]:
    counts: dict[str, dict[str, int]] = {cls: {} for cls in CLASSES}
    images: dict[str, int] = {}
    for row in load_index(root):
        split = row.get("split", "?")
        images[split] = images.get(split, 0) + 1
        class_counts = row.get("class_counts")
        pairs = class_counts or Counter(b["cls"] for b in row_boxes(root, row))
        for cls, n in pairs.items():
            if cls in counts:
                counts[cls][split] = counts[cls].get(split, 0) + int(n)
    table = [["class", "train", "val", "test", "total"]]
    for cls in CLASSES:
        per = counts[cls]
        table.append([cls, *(str(per.get(s, 0)) for s in ("train", "val", "test")), str(sum(per.values()))])
    table.append(["images", *(str(images.get(s, 0)) for s in ("train", "val", "test")), str(sum(images.values()))])
    return table


def report_markdown() -> str:
    for candidate in (
        LOCAL_OUT / "report.md",
        hub_dir() / PREFIX / VERSION / "report.md",
    ):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type=_repo_type(),
            filename=f"{PREFIX}/{VERSION}/report.md",
            token=_token(),
        )
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"_no report available yet ({type(exc).__name__})_"


def build_ui():
    import gradio as gr

    root = dataset_root()
    with gr.Blocks(title="av-datamate dataset viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## av-datamate\nObject-detection dataset for autonomous driving "
            f"(7 canonical classes). Source: `{SOURCE_REPO}` / `{PREFIX}/{VERSION}`, "
            f"served from `{root}`."
        )
        with gr.Row():
            split_box = gr.Dropdown(["train", "val", "test"], value="train", label="Split")
            class_box = gr.Dropdown(["all", *CLASSES], value="car", label="Class filter")
            limit_box = gr.Slider(1, 48, value=MAX_GALLERY, step=1, label="Images")
            reload = gr.Button("Refresh", variant="primary")
        gallery = gr.Gallery(label="Images with bounding boxes", columns=4, object_fit="contain", height=620)
        with gr.Row():
            table = gr.Dataframe(
                headers=["class", "train", "val", "test", "total"],
                label="Class distribution",
                wrap=True,
            )
        gr.Markdown("### Quality report")
        report_box = gr.Markdown()

        def refresh(split, cls, limit):
            return sample_gallery(root, split, cls, int(limit))

        def full_report():
            return report_markdown()

        reload.click(refresh, [split_box, class_box, limit_box], gallery)
        split_box.change(refresh, [split_box, class_box, limit_box], gallery)
        class_box.change(refresh, [split_box, class_box, limit_box], gallery)
        limit_box.change(refresh, [split_box, class_box, limit_box], gallery)
        demo.load(lambda: stats_table(root), [], table)
        demo.load(full_report, [], report_box)
        demo.load(lambda: sample_gallery(root, "train", "car", 12), [], gallery)
    return demo


def main() -> None:
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, quiet=True)


if __name__ == "__main__":
    main()
