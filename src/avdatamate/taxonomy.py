"""Canonical label taxonomy shared by every pipeline stage."""

from __future__ import annotations

CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def is_target(name: str) -> bool:
    return name in CLASS_TO_IDX


def normalize_name(name: str) -> str:
    key = " ".join(str(name).strip().lower().split())
    return key.replace("-", "_").replace(" ", "_")


CLASS_ALIASES: dict[str, str] = {
    "truck": "truck",
    "lorry": "truck",
    "person": "pedestrian",
    "bicycle": "cyclist",
    "bike": "cyclist",
    "motorcycle": "cyclist",
    "motorbike": "cyclist",
    "2_wheeler": "cyclist",
    "3_wheeler": "car",
    "4_wheeler": "car",
    "van": "car",
    "automobile": "car",
    "traffic_light": "traffic_light",
    "trafficlight": "traffic_light",
    "traffic_sign": "traffic_sign",
    "trafficsign": "traffic_sign",
    "stop_sign": "traffic_sign",
}


def resolve_class(name: str) -> str | None:
    """Map a free-form label name onto the canonical taxonomy."""
    key = normalize_name(name)
    if key in CLASS_TO_IDX:
        return key
    return CLASS_ALIASES.get(key)
