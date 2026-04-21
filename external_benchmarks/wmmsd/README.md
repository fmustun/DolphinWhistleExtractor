# WMMSD Benchmark

This folder contains a clean, isolated workflow to benchmark `dolphinteam/DolphinWhistle-CNN-VGG16` on the Watkins Marine Mammal Sound Database (WMMSD) without touching the main training or inference code paths.

## Goal

The first WMMSD pass is intentionally simple:

- stage a flat directory of selected WMMSD clips with collision-safe filenames
- run the existing whistle detector unchanged
- compute a `positive-only` summary at clip level

`positive-only` means:

- a clip is counted as a hit if the CNN emits at least one positive detection
- this is useful for fast cross-dataset screening
- this is **not** a strict whistle recall metric, because the WMMSD selection here is species-level, not whistle-level

## Why The Flat Staging Step Exists

The current inference pipeline writes outputs by clip stem. If we feed it nested WMMSD folders directly, two different files with the same stem can collide in the output tree.

`prepare_wmmsd_manifest.py` avoids that by generating unique names like:

```text
00001__Bottlenose_Dolphin__example__9f0c2f91d3.wav
```

## Data Sources

As checked on April 19, 2026, the official WHOI WMMSD site was redirecting to a maintenance page. For that reason the preparation script supports two input modes:

1. local WMMSD folder tree
2. unofficial Hugging Face mirror such as `confit/wmms-parquet`

If you use WMMSD in publications or shared reports, keep the original credit line:

> "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum."

## Recommended First Pass

Start with:

- `Bottlenose_Dolphin`
- `Common_Dolphin`

That gives us a first read on whether the CNN fires on external delphinid clips before we try to refine the selection toward whistle-specific subsets.

## Commands

### 1. Prepare from a local WMMSD tree

```bash
python external_benchmarks/wmmsd/prepare_wmmsd_manifest.py \
  --source-root /path/to/WMMSD \
  --output-dir benchmark_data/wmmsd_first_pass \
  --species Bottlenose_Dolphin Common_Dolphin
```

### 2. Prepare from the Hugging Face mirror

```bash
python external_benchmarks/wmmsd/prepare_wmmsd_manifest.py \
  --hf-dataset-id confit/wmms-parquet \
  --output-dir benchmark_data/wmmsd_first_pass \
  --species Bottlenose_Dolphin Common_Dolphin
```

### 3. Run inference

```bash
python external_benchmarks/wmmsd/run_wmmsd_inference.py \
  --prepared-dir benchmark_data/wmmsd_first_pass \
  --output-dir benchmark_data/wmmsd_first_pass/inference \
  --model-path models/run3/model_vgg_best.pt \
  --threshold 0.5 \
  --batch-size 64 \
  --max-workers 4
```

### 4. Evaluate positive-only hit rate

```bash
python external_benchmarks/wmmsd/evaluate_wmmsd_positive_only.py \
  --prepared-dir benchmark_data/wmmsd_first_pass \
  --inference-dir benchmark_data/wmmsd_first_pass/inference
```

## Output Structure

After preparation:

```text
benchmark_data/wmmsd_first_pass/
  flat_recordings/
  manifest_all.csv
  manifest_selected.csv
  unsupported_files.csv
  specific_files.txt
  prepare_summary.json
```

After evaluation:

```text
benchmark_data/wmmsd_first_pass/inference/_wmmsd_positive_only/
  per_clip.csv
  per_species.csv
  summary.json
```

## Interpretation Caveat

WMMSD is a reference archive, not a ready-made `whistle vs noise` benchmark in the same format as your local test set. So for this first pass:

- `clip_hit_rate` tells us whether the model fires on external clips from the target species
- it does **not** tell us whether every detected event is a true whistle
- it does **not** yet measure false positives on dedicated non-whistle clips

That is a good first checkpoint, and then we can tighten the protocol once we inspect the WMMSD subset and its metadata more closely.
