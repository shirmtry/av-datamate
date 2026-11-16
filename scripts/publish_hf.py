#!/usr/bin/env python3
"""Publish av-datamate to the Hugging Face hub: Docker Space + processed dataset + report.

Steps (idempotent, never deletes existing hub files):
  a) create the docker Space
  b) upload the app code + Dockerfile + generated Space README (with front matter)
  c) upload the processed dataset + report into the target repo under av-datamate/<version>/
  d) poll the Space runtime until RUNNING

Usage:
  HF_TOKEN=... python scripts/publish_hf.py --version v1
  python scripts/publish_hf.py --code-only          # Space refresh only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from huggingface_hub import HfApi  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402

from avdatamate.config import Layout, PipelineConfig  # noqa: E402
from avdatamate.publish import (  # noqa: E402
    DATASET_DIR_IN_REPO,
    detect_repo_type,
    load_token_file,
    publish_dataset,
)

DEFAULT_SPACE = "Huy1223/av-datamate"
DEFAULT_DATASET_REPO = "Huy1223/datamate_models"
APP_PORT = 7860

CODE_PATTERNS = [
    "Dockerfile",
    "docker-entrypoint.sh",
    ".dockerignore",
    "requirements.txt",
    "pyproject.toml",
    "LICENSE",
    "DATASET_CARD.md",
    "THIRD_PARTY.md",
    "app/*",
    "app/**/*",
    "src/*",
    "src/**/*",
    "configs/*",
    "configs/**/*",
    "scripts/publish_hf.py",
]


def space_readme(space_title: str, short_description: str, license_name: str) -> str:
    """README with the front matter HF needs to build a Docker Space (GitHub README stays clean)."""
    return f"""---
title: {space_title}
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: {APP_PORT}
app_file: app/app.py
pinned: false
license: {license_name}
short_description: {short_description}
tags:
  - autonomous-driving
  - object-detection
  - yolo
  - dataset
---

# {space_title}

Dataset quality viewer for the av-datamate pipeline: browse the processed
autonomous-driving images with their YOLO bounding boxes and read the
per-split / per-class statistics.

The dataset itself is published under `{DATASET_DIR_IN_REPO}/<version>/` in
`{DEFAULT_DATASET_REPO}`; the pipeline source lives at
[github.com/shirmtry/av-datamate](https://github.com/shirmtry/av-datamate).
"""


def upload_space_code(api: HfApi, space: str, version: str, logger=print) -> None:
    readme = space_readme(
        space_title="av-datamate dataset viewer",
        short_description="YOLO dataset viewer for autonomous driving",
        license_name="mit",
    )
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=str(REPO_ROOT),
        allow_patterns=CODE_PATTERNS,
        ignore_patterns=["out/*", "out/**/*", "cache/*", "cache/**/*", ".scratch/*", ".scratch/**/*",
                          "tests/*", "tests/**/*", "third_party/*", "third_party/**/*",
                          ".github/*", ".github/**/*", "README.md", "requirements-dev.txt"],
        commit_message=f"deploy av-datamate app (dataset {version})",
        revision="main",
    )
    # The Space reads its SDK metadata from README.md; the GitHub README stays front-matter free.
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        revision="main",
        commit_message=f"space metadata (dataset {version})",
    )
    logger(f"uploaded code + README -> https://huggingface.co/spaces/{space}")


def poll_runtime(api: HfApi, space: str, timeout: int, logger=print) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            runtime = api.get_space_runtime(space)
        except Exception as exc:
            logger(f"runtime check failed: {type(exc).__name__}: {exc}")
            time.sleep(20)
            continue
        stage = str(runtime.stage)
        if stage != last:
            logger(f"space stage: {stage}")
            last = stage
        if stage == "RUNNING":
            return stage
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "PAUSED", "NO_APP_FILE", "CONFIG_ERROR"}:
            dump_build_hint(api, space, stage, logger)
            return stage
        time.sleep(15)
    return last or "TIMEOUT"


def dump_build_hint(api: HfApi, space: str, stage: str, logger=print) -> None:
    """Best-effort build log fetch; the SSE log endpoint is not always available."""
    logger(f"space reached {stage}; trying to read the build log")
    try:
        import requests

        token = os.environ.get("HF_TOKEN", "")
        resp = requests.get(
            f"https://huggingface.co/api/spaces/{space}/runtime/logs/build",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=30,
            stream=False,
        )
        text = resp.text.strip()
        if text:
            logger(text[-3000:])
        else:
            logger(f"empty build log (HTTP {resp.status_code}); inspect {space} on the hub")
    except Exception as exc:
        logger(f"could not fetch build log ({type(exc).__name__}); inspect the Space page")


def create_space(api: HfApi, space: str, logger=print) -> bool:
    """True when the Space exists. HF answers 402 for free cpu-basic Docker Spaces without PRO."""
    try:
        api.create_repo(
            repo_id=space,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status != 402:
            raise
        logger(
            f"SPACE SKIPPED: {space} cannot be created - Hugging Face requires a PRO subscription "
            "for free cpu-basic Docker Spaces (https://huggingface.co/pro). Dataset publish continues."
        )
        return False
    logger(f"space ready: {space}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sources.yaml")
    parser.add_argument("--code-only", action="store_true", help="skip dataset upload")
    parser.add_argument("--skip-space", action="store_true", help="only upload the dataset")
    parser.add_argument("--timeout", type=int, default=1200, help="runtime polling timeout (s)")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--token-file", type=Path, help="read the token from this file into HF_TOKEN (value never printed)")
    args = parser.parse_args(argv)

    if args.token_file:
        load_token_file(args.token_file)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set - refusing to continue", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    me = api.whoami()
    print(f"authenticated as {me['name']} (token not printed)")

    space_ready = not args.skip_space and create_space(api, args.space)
    if space_ready:
        upload_space_code(api, args.space, args.version)

    published = False
    if not args.code_only:
        cfg = PipelineConfig.load(args.config)
        layout = Layout(args.out)
        repo_type = detect_repo_type(api, args.dataset_repo)
        print(f"{args.dataset_repo} is repo_type={repo_type}")
        result = publish_dataset(cfg, layout, args.dataset_repo, args.version, token=token)
        published = True
        print(f"dataset published: {result['files']} files -> {result['url']}")

    if args.skip_space:
        space_state = "skipped (--skip-space)"
    elif space_ready:
        space_state = "ready"
    else:
        space_state = "NOT DEPLOYED - see the billing message above"
    print(f"summary: space={space_state}, dataset={'published' if published else 'skipped'}")
    if space_ready and not args.no_wait:
        stage = poll_runtime(api, args.space, args.timeout)
        print(f"final space stage: {stage}")
        return 0 if stage == "RUNNING" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
