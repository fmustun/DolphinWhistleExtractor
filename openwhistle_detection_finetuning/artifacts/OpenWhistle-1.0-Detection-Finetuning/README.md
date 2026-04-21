---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# OpenWhistle-1.0-Detection-Finetuning

Binary whistle-vs-noise finetuning dataset rebuilt from `dolphinteam/DophinWhistle-Detection-Finetuning_OLD`.

## Source

- Source repo: `dolphinteam/DophinWhistle-Detection-Finetuning_OLD`
- Build rule: collapse every non-`noise` legacy label into `whistle`
- Split rule: `grouped_by_source_provenance`
- Test size: `0.2`
- Seed: `42`

## Features

- `audio`: audio clips stored with `decode=False`
- `label`: binary class label with values `noise` and `whistle`
- `name`: original clip filename from the legacy dataset
- `source_label`: original legacy label before the binary collapse

## Rows By Split

- `train`: `5527` rows
- `test`: `1385` rows

## Binary Label Counts

- `total`: `noise=2912, whistle=4000`
- `train`: `noise=2109, whistle=3418`
- `test`: `noise=803, whistle=582`

## Source Label Counts

- `train`: `Luna=1456, Nana=145, Neo=1444, Nikita=329, Shy=44, noise=2109`
- `test`: `Dana=130, Yosefa=452, noise=803`

## Source Group Counts

- `train`: `identity::Luna=1456, identity::Nana=145, identity::Neo=1444, identity::Nikita=329, identity::Shy=44, noise_family::noise_chunk=2109`
- `test`: `identity::Dana=130, identity::Yosefa=452, noise_family::new_noise_chunk=803`

## Example

```python
from datasets import Audio, load_dataset

dataset = load_dataset("dolphinteam/OpenWhistle-1.0-Detection-Finetuning")
decoded_train = dataset["train"].cast_column("audio", Audio())
sample = decoded_train[0]
```
