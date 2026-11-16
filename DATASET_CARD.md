# av-datamate — dataset card

Processed YOLO-format object-detection dataset for autonomous driving, built by
the [`av-datamate`](https://github.com/shirmtry/av-datamate) pipeline from
publicly licensed detection datasets on the Hugging Face hub.

* **Task**: 2D bounding-box object detection (YOLOv5/v8 text label format)
* **Canonical classes (7, fixed order)**: `car truck bus pedestrian cyclist traffic_light traffic_sign`
* **Publisher**: Huy1223 (automated pipeline run, seed `1337`)
* **Code licence**: MIT — see [`LICENSE`](LICENSE)
* **Data licence**: per source, see the table below. The result is a
  multi-license corpus: each image keeps the licence of its source, and both CC
  BY 4.0 sources require attribution (given below).
* **Where it lives**: `av-datamate/<version>/` inside
  [`Huy1223/datamate_models`](https://huggingface.co/Huy1223/datamate_models)

## Contents of a version folder

```
av-datamate/v1/
├── dataset/
│   ├── data.yaml                  # path/train/val/test + nc=7 + names
│   ├── images/{train,val,test}/   # <uid>.jpg
│   └── labels/{train,val,test}/   # <uid>.txt, one "cls cx cy w h" per box, normalised
├── splits.jsonl                   # one row per published image: uid, source, group, split, box list
├── report.json                    # machine-readable quality report
├── report.md                      # human-readable quality report
├── validated.jsonl                # samples that passed validation (provenance)
└── rejected.jsonl                 # samples dropped, with the reason
```

`<uid>` is `<source-slug>__<sanitised upstream file stem>`, so every published
image can be traced back to one file in one upstream repo.

## Sources

| Source repo | Type | Licence | Upstream size | Used here | Class mapping |
| --- | --- | --- | --- | --- | --- |
| [`thundarstrom/traffic-vehicle-detection`](https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection) | dataset | CC BY 4.0 (front matter of its `README.md`; `README.dataset.txt` states "License: CC BY 4.0" for the Roboflow UA-DETRAC sample) | 23,319 CCTV images, 215k boxes | 134 images (v1 run) | `car→car`, `bus→bus`, `truck→truck`, `van→car` |
| [`Abtinzandi/Obstacle-Detection-Dataset-YOLO`](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO) | dataset | MIT (front matter of its `README.md`) | 24,326 images, 25 classes | 48 images (v1 run) | `Car→car`, `Truck→truck`, `Bus→bus`, `Van→car`, `Person→pedestrian`, `Bike`/`Motorcycle`/`2-wheeler→cyclist`, `Traffic sign→traffic_sign`, `Traffic light→traffic_light`; the other 17 classes (Building, Road, Tree, Bench, …) are **not** in the taxonomy, so their boxes are dropped |
| [`visaitech/vehicle-mixed-traffic-detection`](https://huggingface.co/datasets/visaitech/vehicle-mixed-traffic-detection) | dataset | CC BY 4.0 (front matter of its `README.md`) | 293 annotated dashcam frames from 6 clips | 133 images (v1 run, all 6 clips) | `pedestrian→pedestrian`, `2-wheeler→cyclist`, `3-wheeler→car`, `4-wheeler→car` |

Sources deliberately excluded because their licence does not allow this kind of
re-distribution: **nuScenes, Waymo Open, KITTI, Cityscapes, BDD100K**.

### Attribution (required by the CC BY 4.0 sources)

> Contains material from *Edge-AI Traffic & Vehicle Analytics System* by
> `thundarstrom` (UA-DETRAC CCTV frames prepared on Roboflow),
> https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection —
> licensed CC BY 4.0.
>
> Contains material from *Visaitech Mixed-Traffic Vehicle Detection Dataset* by
> `visaitech`,
> https://huggingface.co/datasets/visaitech/vehicle-mixed-traffic-detection —
> licensed CC BY 4.0.

Modifications: images were re-named to stable `uid`s, filtered (see Quality
controls) and their boxes re-indexed onto the 7-class taxonomy.

## Split policy

80 / 10 / 10 (train / val / test), **group-aware**: every image inherits the
scene group of its upstream file name — `MVI_<clip>` for the CCTV source, the
10-frame burst prefix of `IMG_<seq>` for ROD, `V<video>` for the dashcam set —
and a whole group is assigned to exactly one split. Ordering is
`sha256("<seed>:<group>")`, so a re-run with the same seed reproduces the split
byte for byte. The pipeline aborts if any group appears in two splits.

Because groups are atomic, val/test sizes are quantised by clip length; with few
clips a split can deviate from the nominal 10%.

## Quality controls

Samples are dropped (and listed in `rejected.jsonl` with a reason) when:
`corrupt_image` (fails PIL verify), `duplicate_image` (SHA-256 of pixels/content
already seen), `image_too_small` (shortest side < 32 px), `missing_label`,
`empty_label` or `no_valid_boxes`.

Individual boxes are dropped when: `malformed_label`, `unknown_class_index`,
`unmapped_class` (not in the 7-class taxonomy), `bbox_out_of_bounds` (centre or
extent outside [0,1]), `bbox_degenerate` (side < 0.2 % of the image, area
< 25 px²).

## Current numbers

See [`report.md`](out/report.md) produced by the same run (also uploaded next to
the dataset). Summary of the published version:

<!--NUMBERS-->

| metric | value |
| --- | --- |
| version | `v1` (seed 1337, pipeline run 2026-11-17) |
| ingested / validated / published | 400 / 315 / 315 images |
| rejected | 85 (`no_valid_boxes` 83, `empty_label` 2) |
| boxes | 2,061 |
| splits | train 266 / val 20 / test 29 images over 31 scene groups |
| class distribution (boxes) | car 406, truck 84, bus 1016, pedestrian 85, cyclist 470, traffic_light 0, traffic_sign 0 |
| images per source | thundarstrom 134, Abtinzandi 48, visaitech 133 |
| resolution | 640×640 (CCTV source) and 1280×720 (dashcam sources) |
| payload size | ~22 MB for 315 images + labels |

## Known limitations

* **`traffic_light` and `traffic_sign` are empty.** The permissive detection
  corpora that annotate them (LISA is CC BY-NC-SA, Charlotte/Korean SETs are far
  above this pipeline's 500-image budget) cannot be re-distributed, so no source
  in `configs/sources.yaml` supplies them. The classes are kept in the taxonomy
  and in `data.yaml` so the dataset stays forward-compatible: adding a licensed
  source needs a config entry only, no code change.
* **Imbalanced class mix.** Vehicle classes dominate because the two largest
  sources are vehicle datasets; ROD contributes most `cyclist` boxes.
* **Small corpus.** The published artifact is deliberately a few hundred images
  (the pipeline can scale to the ~27k permissively licensed upstream images by
  raising `--max-samples`, but the CI/Space budget stays at ≤500).
* **2D boxes only**, single camera views, no depth, no tracking ids, no
  instance masks.

## How to use

```python
from huggingface_hub import snapshot_download

root = snapshot_download(
    repo_id="Huy1223/datamate_models",
    allow_patterns=["av-datamate/v1/**"],
    local_dir="data",
)
# then point any YOLO trainer at data/av-datamate/v1/dataset/data.yaml
```

## Reproduce

```bash
git clone https://github.com/shirmtry/av-datamate && cd av-datamate
pip install -r requirements.txt
export HF_TOKEN=...
python -m avdatamate --out out run --max-samples 400
```

## Citation

```bibtex
@software{av_datamate_2026,
  title  = {av-datamate: a licensed AV detection dataset pipeline},
  author = {Huy1223},
  year   = {2026},
  url    = {https://github.com/shirmtry/av-datamate}
}
```
