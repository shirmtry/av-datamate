# av-datamate

Autonomous-driving dataset pipeline: pick permissively licensed detection
datasets on the Hugging Face hub, **ingest → validate → convert to YOLO →
split → report → publish**, plus a Gradio viewer that ships as a Docker image.

Target task: 2D object detection with a fixed 7-class taxonomy —
`car, truck, bus, pedestrian, cyclist, traffic_light, traffic_sign`.

```
raw downloads          validated samples           publishable dataset
out/raw/images   ─┐    out/validated.jsonl    ┌─►   out/dataset/images/{train,val,test}
out/raw/labels   ─┴─►  out/rejected.jsonl   ──┤     out/dataset/labels/{train,val,test}
                                               ├─►   out/dataset/data.yaml
                                               └─►   out/report.json, out/report.md
```

## Quickstart

```bash
pip install -r requirements.txt
export HF_TOKEN=...           # optional, only for gated sources
python -m avdatamate --token-file ../HF.txt --out out run --max-samples 500
```

`--token-file` reads the token, strips BOM/whitespace and puts it in the
`HF_TOKEN` environment variable of the current process. Nothing ever prints or
commits it (`HF.txt`, `.env`, `*.token` are git-ignored).

Then point a viewer at the output:

```bash
pip install gradio==5.50.0
python app/app.py            # http://127.0.0.1:7860
```

## Pipeline stages

| stage | module | what it guarantees |
| --- | --- | --- |
| ingest | [`src/avdatamate/ingest.py`](src/avdatamate/ingest.py) | lists each source once (cached), honours `--max-samples` with an even quota per source, resumes from `out/raw/manifest.jsonl` without re-downloading |
| validate | [`src/avdatamate/validate.py`](src/avdatamate/validate.py) | drops corrupt images, SHA-256 duplicates, images below 32 px, missing labels; drops boxes that are malformed, out of frame, degenerate or outside the canonical taxonomy |
| convert | [`src/avdatamate/convert.py`](src/avdatamate/convert.py) | writes YOLO `images/`, `labels/` and `data.yaml` with canonical class ids |
| split | [`src/avdatamate/split.py`](src/avdatamate/split.py) | 80/10/10 by **scene group** (camera clip / burst), deterministic in the seed, refuses to publish if any group straddles two splits |
| report | [`src/avdatamate/report.py`](src/avdatamate/report.py) | `report.json` + human-readable `report.md`: per-stage counts, class distribution, image sizes, every rejection reason, licence per source |
| publish | [`src/avdatamate/publish.py`](src/avdatamate/publish.py) | uploads only `dataset/` + reports under `av-datamate/<version>/`, never the raw downloads, never touching other paths in the target repo |

CLI (each stage is also runnable on its own):

```bash
avdatamate --out out run      --max-samples 500   # full pipeline
avdatamate --out out ingest   --max-samples 500
avdatamate --out out validate
avdatamate --out out convert
avdatamate --out out split    --seed 1337
avdatamate --out out report
avdatamate --out out publish  --target-repo Huy1223/datamate_models --version v1 [--dry-run]
```

## Sources and licences

Only re-distributable licences are used, and `configs/sources.yaml` is the
single place where a source is declared. `tests/test_taxonomy.py` fails the
build if a source is missing a licence or uses one outside the allow-list, or if
a `class_map` targets a non-canonical class. See
[`DATASET_CARD.md`](DATASET_CARD.md) for the full card and
[`THIRD_PARTY.md`](THIRD_PARTY.md) for reference code and dependencies.

Deliberately **not** used: nuScenes, Waymo Open, KITTI, Cityscapes, BDD100K —
their licences prohibit this kind of re-distribution.

## Docker

```bash
docker build -t av-datamate .
docker run --rm -p 7860:7860 av-datamate                       # viewer
docker run --rm av-datamate avdatamate --out out run --max-samples 50
```

Base image `python:3.11-slim`, CPU only, pinned dependencies, runs as UID 1000,
exposes 7860.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The suite runs against a synthetic YOLO repository on disk
([`tests/conftest.py`](tests/conftest.py)) — no network, no images committed,
35 samples instead of thousands.

## Known limitations

* `traffic_light` and `traffic_sign` come out empty from the current sources:
  the permissive detection sets that cover them (LISA, Charlotte) are either
  non-commercial or too large for this pipeline's budget. The classes stay in
  the taxonomy so the dataset is forward-compatible.
* Split sizes are bounded by the number of scene groups: with few clips per
  source, val/test can be much smaller than 10%.
