# Adriatic Benchmark

This folder contains a clean, isolated workflow to benchmark `dolphinteam/DolphinWhistle-CNN-VGG16` on the Adriatic trawling dataset without touching the main training or inference code paths.

## Goal

Unlike the WMMSD proxy benchmark, this Adriatic workflow builds a true clip-level `whistle vs noise` test set from temporal annotations:

- positives are `0.4 s` windows centered on whistle annotations from `whistles.txt`
- negatives are `0.4 s` windows mined from `full_recording.wav` away from whistles and, by default, away from click trains too
- the prepared clips are then passed through the existing whistle detector unchanged

## Expected Source Files

The preparation script expects a dataset root containing, somewhere in the tree:

- `full_recording.wav`
- `whistles.txt`
- `clicks.txt` (optional, but recommended)

It can also work directly from a downloaded archive via `--archive-path`.

According to the associated data paper, `full_recording.wav` is a `192 kHz` raw recording of `98 min 20 s`, and `whistles.txt` contains `303` whistle annotations with tag (`W` or `MW`) and quality (`1` to `3`, where quality increases from `1` to `3`).

Sources:

- https://www.nature.com/articles/s41597-023-02547-8
- https://figshare.com/articles/dataset/Detection_and_classification_of_bottlenose_dolphin_s_Tursiops_truncatus_Montagu_1821_vocalizations_in_the_central-northern_Adriatic_Sea/21621222

## Recommended First Pass

Start with:

- `window_seconds = 0.4`
- `margin_seconds = 0.4`
- `noise_to_whistle_ratio = 1.0`
- keep `MW` annotations
- keep click trains excluded from negative mining

That gives a conservative whistle/noise benchmark aligned with the current CNN task.

## Commands

### 1. Prepare the clip-level benchmark

Using a dataset root:

```bash
python external_benchmarks/adriatic/prepare_adriatic_clip_benchmark.py \
  --source-root /path/to/adriatic_dataset \
  --output-dir benchmark_data/adriatic_clip \
  --window-seconds 0.4 \
  --margin-seconds 0.4 \
  --noise-to-whistle-ratio 1.0 \
  --min-whistle-quality 1
```

Using explicit file paths:

```bash
python external_benchmarks/adriatic/prepare_adriatic_clip_benchmark.py \
  --full-recording-path /path/to/full_recording.wav \
  --whistles-path /path/to/whistles.txt \
  --clicks-path /path/to/clicks.txt \
  --output-dir benchmark_data/adriatic_clip \
  --window-seconds 0.4 \
  --margin-seconds 0.4 \
  --noise-to-whistle-ratio 1.0 \
  --min-whistle-quality 1
```

Using a downloaded archive:

```bash
python external_benchmarks/adriatic/prepare_adriatic_clip_benchmark.py \
  --archive-path /path/to/adriatic_dataset.zip \
  --output-dir benchmark_data/adriatic_clip \
  --window-seconds 0.4 \
  --margin-seconds 0.4 \
  --noise-to-whistle-ratio 1.0 \
  --min-whistle-quality 1
```

### 2. Run inference

```bash
python external_benchmarks/adriatic/run_adriatic_inference.py \
  --prepared-dir benchmark_data/adriatic_clip \
  --output-dir benchmark_data/adriatic_clip/inference \
  --model-path models/run3/model_vgg_best.pt \
  --threshold 0.5 \
  --batch-size 64 \
  --max-workers 1
```

### 3. Evaluate clip-level metrics

```bash
python external_benchmarks/adriatic/evaluate_adriatic_clip_metrics.py \
  --prepared-dir benchmark_data/adriatic_clip \
  --inference-dir benchmark_data/adriatic_clip/inference \
  --clip-positive-min-detections 1 2 3
```

### 4. One-shot runner

```bash
python external_benchmarks/adriatic/run_adriatic_clip_benchmark.py \
  --source-root /path/to/adriatic_dataset \
  --output-dir benchmark_data/adriatic_clip \
  --model-path models/run3/model_vgg_best.pt \
  --batch-size 64 \
  --max-workers 1 \
  --clip-positive-min-detections 1 2 3
```

Or directly from the archive:

```bash
python external_benchmarks/adriatic/run_adriatic_clip_benchmark.py \
  --archive-path /path/to/adriatic_dataset.zip \
  --output-dir benchmark_data/adriatic_clip \
  --model-path models/run3/model_vgg_best.pt \
  --batch-size 64 \
  --max-workers 1 \
  --clip-positive-min-detections 1 2 3
```

## Output Structure

After preparation:

```text
benchmark_data/adriatic_clip/
  flat_recordings/
  manifest_selected.csv
  prepare_summary.json
  README.md
```

After evaluation:

```text
benchmark_data/adriatic_clip/inference/_adriatic_clip_metrics/
  summary.csv
  confusion_matrix.csv
  per_clip.csv
  positive_group_summary.csv
  summary.json
```

## Interpretation Caveat

This benchmark is much closer to the real CNN task than WMMSD, but it is still clip-level:

- one centered `0.4 s` window per whistle annotation
- negatives are mined from clear background regions
- a clip is counted as positive if the CNN emits at least `N` detections, where `N` is the vote threshold used during evaluation

So this is a strong external `whistle vs noise` benchmark, but not yet a fine-grained event-matching benchmark against every whistle onset/offset frame in the full recording.
