# OpenWhistle Detection Finetuning

This folder rebuilds `dolphinteam/DophinWhistle-Detection-Finetuning_OLD`
into a new local `DatasetDict` for
`dolphinteam/OpenWhistle-1.0-Detection-Finetuning`.

The legacy dataset does not expose a reusable train/test split, so the build
script creates a deterministic split stratified by the original `source_label`
values from `trials.csv`, then collapses those source labels into a binary
`noise` / `whistle` label for the final dataset.

## Build locally

```bash
python openwhistle_detection_finetuning/build_openwhistle_detection_finetuning.py \
  --overwrite
```

Smoke test:

```bash
python openwhistle_detection_finetuning/build_openwhistle_detection_finetuning.py \
  --overwrite \
  --limit 32
```

Useful knobs:

- `--test-size 0.2`
- `--seed 42`
- `--split-policy grouped_rigorous`
- `--source-snapshot-dir /path/to/legacy/snapshot`
- `--output-dir /path/to/local_dataset`
- `--artifacts-dir /path/to/artifacts`

## Push manually

Run this yourself when you are ready:

```bash
python openwhistle_detection_finetuning/push_openwhistle_detection_finetuning.py \
  --repo-id dolphinteam/OpenWhistle-1.0-Detection-Finetuning
```

You can add:

- `--token ...`
- `--private`
- `--max-shard-size 500MB`

## Outputs

After the build, you will have:

- a local `DatasetDict` saved with `save_to_disk()`
- `artifacts/README.md`
- `artifacts/summary.json`
- `artifacts/split_manifest.csv`

The push script uploads the dataset plus those artifacts, but it does not run
unless you call it yourself.
