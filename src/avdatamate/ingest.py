"""Stage 1 - download annotation pairs from the sources declared in configs/sources.yaml."""

from __future__ import annotations

import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .config import IMAGE_EXTS, Layout, PipelineConfig, SourceConfig, append_jsonl

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
DOWNLOAD_WORKERS = 6


def _fetch_pair(
    downloader, source: SourceConfig, item: tuple[str, str, str]
) -> tuple[Path, Path]:
    """Resolve one (uid, image, label) entry to local paths."""
    _, image_rel, label_rel = item
    return (
        Path(downloader.fetch(source.id, image_rel, source.repo_type)),
        Path(downloader.fetch(source.id, label_rel, source.repo_type)),
    )


def log_noop(msg: str) -> None:  # pragma: no cover - default logger
    print(msg, flush=True)


def uid_for(source: SourceConfig, repo_path: str) -> str:
    stem = SafeStem.of(repo_path)
    return f"{source.slug}__{stem}"


class SafeStem:
    @staticmethod
    def of(repo_path: str) -> str:
        name = Path(repo_path).name
        stem = name[: -len(Path(name).suffix)] if Path(name).suffix else name
        stem = SAFE_NAME.sub("_", stem).strip("_")
        return stem[:110] or "sample"


def label_path_for(image_path: str) -> str:
    """YOLO convention: swap the last `images` path segment for `labels`."""
    parts = image_path.split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            parts[-1] = Path(parts[-1]).stem + ".txt"
            return "/".join(parts)
    return ""


def group_key(source: SourceConfig, image_path: str) -> str:
    stem = Path(image_path).name
    stem = stem[: -len(Path(stem).suffix)] if Path(stem).suffix else stem
    if source.group_regex:
        m = re.search(source.group_regex, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return stem


def parse_class_names(data_yaml_text: str) -> list[str]:
    doc = yaml.safe_load(data_yaml_text) or {}
    names = doc.get("names")
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        return [str(v) for _, v in items]
    if isinstance(names, list):
        return [str(v) for v in names]
    return []


class HfDownloader:
    """Thin wrapper over huggingface_hub with on-disk listing cache and retries."""

    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path | str | None = None,
        attempts: int = 3,
        sleep: float = 2.0,
    ) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.attempts = attempts
        self.sleep = sleep
        if self.cache_dir:
            (self.cache_dir / "listings").mkdir(parents=True, exist_ok=True)

    def listing(self, repo_id: str, repo_type: str = "dataset") -> list[dict[str, Any]]:
        cache = self.cache_dir / "listings" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', repo_id)}.json" if self.cache_dir else None
        if cache and cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        files: list[dict[str, Any]] = []
        for tree in self._retry(lambda: list(self.api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True))):
            if not hasattr(tree, "size"):
                continue
            files.append({"path": tree.path, "size": int(tree.size or 0)})
        if cache:
            cache.write_text(json.dumps(files), encoding="utf-8")
        return files

    def read_text(self, repo_id: str, path: str, repo_type: str = "dataset") -> str:
        local = self.fetch(repo_id, path, repo_type)
        return Path(local).read_text(encoding="utf-8", errors="replace")

    def fetch(self, repo_id: str, path: str, repo_type: str = "dataset") -> Path:
        return self._retry(
            lambda: self.api.hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type=repo_type,
                cache_dir=str(self.cache_dir / "hf") if self.cache_dir else None,
            )
        )

    def _retry(self, fn: Callable[[], Any]) -> Any:
        last: Exception | None = None
        for i in range(self.attempts):
            try:
                return fn()
            except Exception as exc:  # network/auth errors are retriable
                last = exc
                if i + 1 < self.attempts:
                    time.sleep(self.sleep * (i + 1))
        raise RuntimeError(f"download failed after {self.attempts} attempts: {last}") from last


@dataclass
class IngestResult:
    selected: int
    downloaded: int
    skipped_existing: int
    per_source: dict[str, int]
    class_names: dict[str, list[str]]


def _pick_pairs(
    files: list[dict[str, Any]], source: SourceConfig, by_path: dict[str, dict[str, Any]]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for entry in files:
        path = entry["path"]
        lower = path.lower()
        if not lower.endswith(IMAGE_EXTS):
            continue
        if "/images/" not in f"/{path}" and not lower.startswith("images/"):
            continue
        if any(path.startswith(p) for p in source.exclude_prefixes):
            continue
        label = label_path_for(path)
        if not label or label not in by_path:
            continue
        pairs.append((path, label))
    pairs.sort()
    return pairs


def allocate_quotas(availabilities: dict[str, int], max_samples: int) -> dict[str, int]:
    """Even split first, then round-robin the unused capacity to the sources that have it."""
    ids = list(availabilities)
    if not ids or max_samples <= 0:
        return {i: 0 for i in ids}
    n = len(ids)
    base, rem = divmod(max_samples, n)
    quotas = {i: min(base + (1 if idx < rem else 0), availabilities[i]) for idx, i in enumerate(ids)}
    left = max_samples - sum(quotas.values())
    while left > 0:
        grew = False
        for i in ids:
            if quotas[i] < availabilities[i]:
                quotas[i] += 1
                left -= 1
                grew = True
                if left == 0:
                    break
        if not grew:
            break
    return quotas


def resolve_class_names(
    source: SourceConfig, downloader: HfDownloader, files: list[dict[str, Any]]
) -> list[str]:
    for candidate in ("data.yaml", "dataset/data.yaml"):
        if candidate in {f["path"] for f in files}:
            try:
                names = parse_class_names(downloader.read_text(source.id, candidate, source.repo_type))
            except Exception:
                names = []
            if names:
                return names
    if source.class_names:
        return list(source.class_names)
    classes_txt = {"classes.txt", "labels/classes.txt"}
    for candidate in classes_txt & {f["path"] for f in files}:
        try:
            text = downloader.read_text(source.id, candidate, source.repo_type)
        except Exception:
            continue
        names = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if names:
            return names
    return []


def ingest(
    cfg: PipelineConfig,
    layout: Layout,
    max_samples: int,
    downloader: HfDownloader,
    logger: Callable[[str], None] = log_noop,
) -> IngestResult:
    layout.ensure_dirs()
    done: dict[str, dict[str, Any]] = {}
    if layout.manifest.exists():
        for row in _read_manifest(layout.manifest):
            if (layout.abs(row["image"])).exists() and (layout.abs(row["label"])).exists():
                done[row["uid"]] = row
    for row in done.values():
        logger(f"resume: reuse {row['uid']}")

    class_names: dict[str, list[str]] = {}
    picked: dict[str, list[tuple[str, str]]] = {}
    availabilities: dict[str, int] = {}
    for source in cfg.sources:
        files = downloader.listing(source.id, source.repo_type)
        by_path = {f["path"]: f for f in files}
        pairs = _pick_pairs(files, source, by_path)
        availabilities[source.id] = len(pairs)
        picked[source.id] = pairs
        names = resolve_class_names(source, downloader, files)
        class_names[source.id] = names
        logger(f"{source.id}: {len(pairs)} annotated images available, classes={names}")

    quotas = allocate_quotas(availabilities, max_samples)
    downloaded = skipped = 0
    with layout.manifest.open("a", encoding="utf-8") as fh:
        for source in cfg.sources:
            quota = quotas[source.id]
            taken = 0
            names = class_names[source.id]
            batch: list[tuple[str, str, str]] = []
            for image_rel, label_rel in picked[source.id]:
                if taken >= quota:
                    break
                ident = uid_for(source, image_rel)
                if ident in done:
                    skipped += 1
                    taken += 1
                    continue
                batch.append((ident, image_rel, label_rel))
                taken += 1
            # the hub is latency-bound, not bandwidth-bound: fetch concurrently,
            # then append the manifest in the deterministic (sorted) pair order
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
                fetched = list(pool.map(lambda item: _fetch_pair(downloader, source, item), batch))
            for (ident, image_rel, label_rel), (src_img, src_lbl) in zip(batch, fetched, strict=True):
                image_local = layout.raw_images / f"{ident}{Path(image_rel).suffix.lower()}"
                label_local = layout.raw_labels / f"{ident}.txt"
                shutil.copyfile(src_img, image_local)
                shutil.copyfile(src_lbl, label_local)
                append_jsonl(
                    fh,
                    {
                        "uid": ident,
                        "source": source.id,
                        "repo_image": image_rel,
                        "repo_label": label_rel,
                        "image": layout.rel(image_local),
                        "label": layout.rel(label_local),
                        "group": group_key(source, image_rel),
                        "class_names": names,
                    },
                )
                fh.flush()
                downloaded += 1
            logger(f"{source.id}: quota {quota}, processed {taken}")
    rows = _read_manifest(layout.manifest)
    layout.write_state(
        "ingest",
        {
            "samples": len(rows),
            "downloaded": downloaded,
            "reused": skipped,
            "quotas": quotas,
            "available": availabilities,
            "class_names": class_names,
        },
    )
    return IngestResult(
        selected=sum(quotas.values()),
        downloaded=downloaded,
        skipped_existing=skipped,
        per_source=quotas,
        class_names=class_names,
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    from .config import read_jsonl

    return read_jsonl(path)


def iter_unique(paths: Iterable[str]) -> list[str]:  # pragma: no cover - helper kept for tests
    seen: list[str] = []
    for p in paths:
        if p not in seen:
            seen.append(p)
    return seen
