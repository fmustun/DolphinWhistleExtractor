# DCLDE Benchmark

This folder contains a clean, isolated workflow to benchmark `dolphinteam/DolphinWhistle-CNN-VGG16` on the public NOAA DCLDE 2011 whistle corpus without touching the main training or inference code paths.

## Goal

The original DCLDE task is whistle contour annotation, not binary clip classification. For the current CNN we therefore build a proxy clip-level `whistle vs noise` benchmark:

- positives are `0.4 s` windows centered on analyst-annotated whistle contours from `.ann` files
- negatives are `0.4 s` windows mined from the same recordings away from any annotated contour by a configurable margin
- the prepared clips are then passed through the existing whistle detector unchanged

This is closer to the CNN task than a species benchmark, but it remains an approximation because DCLDE annotations were created for contour extraction, not for exhaustive binary segmentation of all acoustic events.

## Data Source

The public NOAA archive exposes:

- WAV recordings under `development/` and `evaluation/`
- original 2011 `.ann` contour files alongside annotated WAVs
- revised 2025 annotations under `annotations2025/` for part of the development set
- `silbidopy`, a Python reader for the binary `.ann` format

Primary sources:

- https://catalog.data.gov/dataset/dclde-2011-annotated-odontocete-whistles
- https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/MGG/passive_acoustic/iso/xml/DCLDE_2011.xml
- https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o?prefix=dclde/2011/dclde_2011

## Recommended First Pass

For a first external benchmark of the current CNN:

- use `subset = evaluation`
- use `annotation_version = 2011`
- use `window_seconds = 0.4`
- use `margin_seconds = 0.4`
- use `noise_to_whistle_ratio = 1.0`

That gives a clean held-out subset inside DCLDE and avoids mixing original and revised annotation policies.

## Step 0. Download the subset from NOAA

Download annotated evaluation files plus documentation:

```bash
python external_benchmarks/dclde/download_dclde_subset.py \
  --output-dir benchmark_data/dclde_source \
  --subset evaluation \
  --annotated-only \
  --include-docs
```

Optional development download with revised 2025 annotations:

```bash
python external_benchmarks/dclde/download_dclde_subset.py \
  --output-dir benchmark_data/dclde_source_dev \
  --subset development \
  --annotated-only \
  --include-annotations2025 \
  --include-docs \
  --include-software
```

The resulting dataset root will usually be:

```text
benchmark_data/dclde_source/dclde/2011/dclde_2011
```

## Step 1. Prepare the clip-level benchmark

```bash
python external_benchmarks/dclde/prepare_dclde_clip_benchmark.py \
  --source-root benchmark_data/dclde_source \
  --output-dir benchmark_data/dclde_clip \
  --subset evaluation \
  --annotation-version 2011 \
  --window-seconds 0.4 \
  --margin-seconds 0.4 \
  --noise-to-whistle-ratio 1.0
```

## Step 2. Run inference

```bash
python external_benchmarks/dclde/run_dclde_inference.py \
  --prepared-dir benchmark_data/dclde_clip \
  --output-dir benchmark_data/dclde_clip/inference \
  --model-path models/run3/model_vgg_best.pt \
  --threshold 0.5 \
  --batch-size 64 \
  --max-workers 1
```

## Step 3. Evaluate clip-level metrics

```bash
python external_benchmarks/dclde/evaluate_dclde_clip_metrics.py \
  --prepared-dir benchmark_data/dclde_clip \
  --inference-dir benchmark_data/dclde_clip/inference \
  --clip-positive-min-detections 1 2 3
```

## Step 4. One-shot runner

```bash
python external_benchmarks/dclde/run_dclde_clip_benchmark.py \
  --source-root benchmark_data/dclde_source \
  --output-dir benchmark_data/dclde_clip \
  --subset evaluation \
  --annotation-version 2011 \
  --model-path models/run3/model_vgg_best.pt \
  --batch-size 64 \
  --max-workers 1 \
  --clip-positive-min-detections 1 2 3
```

If `benchmark_data/dclde_source` does not exist yet, the runner can download the required NOAA subset first:

```bash
python external_benchmarks/dclde/run_dclde_clip_benchmark.py \
  --source-root benchmark_data/dclde_source \
  --output-dir benchmark_data/dclde_clip \
  --subset evaluation \
  --annotation-version 2011 \
  --model-path models/run3/model_vgg_best.pt \
  --batch-size 64 \
  --max-workers 1 \
  --clip-positive-min-detections 1 2 3 \
  --download-first
```

If a previous partial or old benchmark already exists in `benchmark_data/dclde_clip`, rerun with:

```bash
python external_benchmarks/dclde/run_dclde_clip_benchmark.py \
  --source-root benchmark_data/dclde_source \
  --output-dir benchmark_data/dclde_clip \
  --subset evaluation \
  --annotation-version 2011 \
  --model-path models/run3/model_vgg_best.pt \
  --batch-size 64 \
  --max-workers 1 \
  --clip-positive-min-detections 1 2 3 \
  --download-first \
  --overwrite-output
```

## Output Structure

After preparation:

```text
benchmark_data/dclde_clip/
  flat_recordings/
  manifest_selected.csv
  source_files.csv
  prepare_summary.json
  README.md
```

After evaluation:

```text
benchmark_data/dclde_clip/inference/_dclde_clip_metrics/
  summary.csv
  confusion_matrix.csv
  per_clip.csv
  positive_group_summary.csv
  summary.json
```

## Interpretation Caveat

This benchmark is useful for the CNN task, but it is not the official DCLDE contour-tracking metric:

- positives are centered clip windows, not full contour traces
- negatives are mined from non-overlapping windows that avoid annotated whistles
- weak or skipped whistles in unannotated regions can still exist

So this is a practical external `whistle vs noise` benchmark built from DCLDE, not a replacement for the original time-frequency contour evaluation protocol.
