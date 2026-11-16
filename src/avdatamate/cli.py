"""Command line entry point: `avdatamate run --max-samples 500 --out ./out`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .config import Layout, PipelineConfig, check_python
from .convert import convert
from .ingest import HfDownloader, ingest, log_noop
from .publish import DEFAULT_TARGET_REPO, load_token_file, publish_dataset
from .report import report
from .split import split
from .validate import validate

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sources.yaml"


def _logger(prefix: str):
    def log(msg: str) -> None:
        print(f"[{prefix}] {msg}", flush=True)

    return log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avdatamate", description="AV dataset -> YOLO pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sources yaml")
    parser.add_argument("--out", type=Path, default=Path("out"), help="pipeline output directory")
    parser.add_argument("--cache", type=Path, default=None, help="download cache (default <out>/cache)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path(os.environ["HF_TOKEN_FILE"]) if os.environ.get("HF_TOKEN_FILE") else None,
        help="file holding the hub token; loaded into HF_TOKEN, never printed",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ingest + validate + convert + split + report")
    run.add_argument("--max-samples", type=int, default=500)
    run.add_argument("--publish", action="store_true")
    run.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    run.add_argument("--version", default=None)
    run.add_argument("--dry-run", action="store_true")

    ing = sub.add_parser("ingest", help="download annotation pairs")
    ing.add_argument("--max-samples", type=int, default=500)

    for name, help_text in (
        ("validate", "drop broken/duplicated/invalid samples"),
        ("convert", "write YOLO format dataset"),
        ("report", "write report.json and report.md"),
    ):
        sub.add_parser(name, help=help_text)

    sp = sub.add_parser("split", help="leak-free 80/10/10 split")
    sp.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    pub = sub.add_parser("publish", help="upload dataset + report to the hub")
    pub.add_argument("--target-repo", default=DEFAULT_TARGET_REPO)
    pub.add_argument("--version", default=None)
    pub.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    check_python()
    args = build_parser().parse_args(argv)
    if args.token_file:
        load_token_file(args.token_file)
    log = log_noop if not args.quiet else (lambda msg: None)
    layout = Layout(args.out)
    layout.root.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig.load(args.config)
    cache = args.cache or (layout.root / "cache")
    cache.mkdir(parents=True, exist_ok=True)

    def downloader() -> HfDownloader:
        return HfDownloader(token=os.environ.get("HF_TOKEN") or None, cache_dir=cache)

    if args.command == "run":
        ingest(cfg, layout, args.max_samples, downloader(), _logger("ingest"))
        validate(cfg, layout, _logger("validate"))
        convert(cfg, layout, _logger("convert"))
        split(cfg, layout, args.seed, _logger("split"))
        report(cfg, layout, _logger("report"))
        if args.publish:
            publish_dataset(
                cfg, layout, args.target_repo, args.version, dry_run=args.dry_run,
                logger=_logger("publish"),
            )
        log(f"done: {layout.report_md}")
    elif args.command == "ingest":
        ingest(cfg, layout, args.max_samples, downloader(), _logger("ingest"))
    elif args.command == "validate":
        validate(cfg, layout, _logger("validate"))
    elif args.command == "convert":
        convert(cfg, layout, _logger("convert"))
    elif args.command == "split":
        split(cfg, layout, args.seed, _logger("split"))
    elif args.command == "report":
        report(cfg, layout, _logger("report"))
    elif args.command == "publish":
        publish_dataset(
            cfg, layout, args.target_repo, args.version, dry_run=args.dry_run,
            logger=_logger("publish"),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
