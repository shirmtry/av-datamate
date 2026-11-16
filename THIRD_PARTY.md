# Third-party usage

Everything below is used as **reference or dependency only**. No third-party
source code is copied into this repository, and nothing here is bundled into the
published dataset or the Docker image.

## Reference repositories (read-only, not shipped)

Cloned with `git clone --depth 1` into `third_party/`, which is git-ignored and
excluded from the Docker build context, so these licenses never apply to the
artifact we distribute.

| Repo | Revision | License | Why it is here |
| --- | --- | --- | --- |
| [nutonomy/nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) | `b40adc467b919192899405d9b77871afee8efa07` | Apache-2.0 | Reference for how a large AV detection dataset organises samples, scene groups/clips and label files. Its data licence (nuScenes) is non-commercial, so **no nuScenes data is used**. |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `ef512bfc250a91fa75632efd1f3b7cc87a0cac3e` | AGPL-3.0 | Reference only for the on-disk YOLO detection contract (`images/`, `labels/`, `data.yaml`, `cls cx cy w h` normalised). AGPL code is **not** imported, linked, installed or redistributed. |
| [roboflow/supervision](https://github.com/roboflow/supervision) | `2d613bcd501e632c0b51bb56facd13534d2a7739` | MIT | Reference for bounding-box annotation/visualisation conventions used by the Gradio viewer. Re-implemented from scratch, no code copied. |

> `commaai/comma2k19` (MIT) was tried as a clip-grouping reference, but its tree
> contains directory names with `|`, which cannot be checked out on Windows.
> Clip grouping is instead taken from the `MVI_*` / `IMG_*` / `V*` naming of the
> sources listed in `DATASET_CARD.md`.

## Runtime dependencies

Pinned in `requirements.txt` / `pyproject.toml`.

| Package | Version | License |
| --- | --- | --- |
| `huggingface_hub` | 1.33.0 | Apache-2.0 |
| `Pillow` | 11.3.0 | MIT-CMU (HPND) |
| `PyYAML` | 6.0.3 | MIT |
| `gradio` | 5.50.0 | Apache-2.0 |

Dev-only: `pytest` 8.4.2 (MIT), `ruff` 0.14.14 (MIT).

## Base image

`python:3.11-slim` (Debian). PSF/Debian licenses apply to the image layers only;
no sources are vendored from it.

## Upstream data

The datasets themselves are documented in [`DATASET_CARD.md`](DATASET_CARD.md);
`configs/sources.yaml` records the licence of every source and the pipeline
refuses to ingest a source whose licence is not in the permissive allow-list
(see `tests/test_taxonomy.py`).
