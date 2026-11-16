"""av-datamate: autonomous-driving dataset pipeline (ingest -> validate -> convert -> split -> report -> publish)."""

from .config import Layout, PipelineConfig
from .taxonomy import CLASS_TO_IDX, CLASSES

__all__ = ["CLASSES", "CLASS_TO_IDX", "Layout", "PipelineConfig", "main"]
__version__ = "1.0.0"


def main(argv=None):  # pragma: no cover - thin wrapper
    from .cli import main as _main

    return _main(argv)
