"""Stage 6 - push the processed dataset and quality report to the Hugging Face hub."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .config import Layout, PipelineConfig

DEFAULT_TARGET_REPO = "Huy1223/datamate_models"
DATASET_DIR_IN_REPO = "av-datamate"


class PublishError(RuntimeError):
    pass


def get_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise PublishError("HF_TOKEN environment variable is not set")
    return token


def load_token_file(path: Path | str) -> str:
    """Read a token from a file into the HF_TOKEN env var; the value is never logged."""
    file = Path(path)
    if not file.is_file():
        raise PublishError(f"token file {file} not found")
    text = file.read_bytes().decode("utf-8-sig", errors="replace")
    token = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not token:
        raise PublishError(f"token file {file} is empty")
    os.environ["HF_TOKEN"] = token
    return token


def _api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def detect_repo_type(api, repo_id: str) -> str:
    for repo_type in ("model", "dataset", "space"):
        try:
            api.repo_info(repo_id, repo_type=repo_type)
            return repo_type
        except Exception:
            continue
    raise PublishError(f"{repo_id} not found as model/dataset/space (and cannot be created here)")


def publish_dataset(
    cfg: PipelineConfig,
    layout: Layout,
    repo_id: str = DEFAULT_TARGET_REPO,
    version: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    logger: Callable[[str], None] = lambda m: print(m, flush=True),
) -> dict[str, Any]:
    """Upload dataset/ + reports under av-datamate/<version>/ without touching other paths."""
    if not layout.dataset.is_dir():
        raise PublishError(f"{layout.dataset} missing - run the pipeline first")
    token = token or get_token()
    api = _api(token)
    repo_type = detect_repo_type(api, repo_id)
    version = version or f"v{cfg.version}"
    prefix = f"{DATASET_DIR_IN_REPO}/{version}"

    payload = _collect_payload(layout)
    message = f"av-datamate {version}: {len(payload)} files from pipeline run"
    logger(f"publish -> {repo_id} (repo_type={repo_type}, path_in_repo={prefix})")
    if dry_run:
        return {
            "repo": repo_id,
            "repo_type": repo_type,
            "path_in_repo": prefix,
            "files": len(payload),
            "dry_run": True,
            "url": f"https://huggingface.co/{repo_id}/tree/main/{prefix}",
        }
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(layout.root),
        path_in_repo=prefix,
        allow_patterns=payload,
        commit_message=message,
    )
    result = {
        "repo": repo_id,
        "repo_type": repo_type,
        "path_in_repo": prefix,
        "files": len(payload),
        "commit_message": message,
        "url": f"https://huggingface.co/{repo_id}/tree/main/{prefix}",
    }
    layout.write_state("publish", result)
    (layout.root / "publish.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger(f"published: {result['url']}")
    return result


def _collect_payload(layout: Layout) -> list[str]:
    """Everything under dataset/ plus the reports; raw downloads never leave the machine."""
    payload: list[str] = []
    for base in (layout.dataset_images, layout.dataset_labels):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                payload.append(path.relative_to(layout.root).as_posix())
    for extra in (
        layout.dataset_data_yaml,
        layout.splits_index,
        layout.report_json,
        layout.report_md,
        layout.validated,
        layout.rejected,
    ):
        if extra.is_file():
            payload.append(extra.relative_to(layout.root).as_posix())
    if not payload:
        raise PublishError("nothing to publish - pipeline output is empty")
    return payload
