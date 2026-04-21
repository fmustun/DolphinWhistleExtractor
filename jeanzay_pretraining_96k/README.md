# Jean Zay Pretraining 96 kHz

Ce dossier contient un workflow séparé pour reconstruire `DolphinWhistle-Pretraining-96k` sur Jean Zay sans garder la logique HPC dans le script local principal.

## Ce qu'il contient

- `stage_pretraining_96k_for_jeanzay.py`
  Prépare localement un arbre de staging compact avec seulement les segments et les enregistrements 96 kHz nécessaires.
- `rebuild_pretraining_96k_jeanzay.py`
  Wrapper léger qui charge `scripts/rebuild_pretraining_96k.py` et lui injecte un `path_config.json`.
- `rebuild_pretraining_96k.sbatch`
  Job SLURM CPU prêt à lancer sur Jean Zay.

## Workflow

### 1. Préparer le staging local

Depuis le repo local :

```bash
python jeanzay_pretraining_96k/stage_pretraining_96k_for_jeanzay.py \
  --output-root /home/pablo/jeanzay_pretraining_96k_stage \
  --reset-output
```

Par défaut, le script copie les fichiers dans un arbre portable :

```text
jeanzay_pretraining_96k_stage/
  data/
    recordings/
    segments/
  generated/
    path_config.json
    summary.json
    transfer_manifest.json
```

Si la destination finale est un montage distant comme `/home/pablo/ibens/...`, prefere l'archive unique plutot que la copie fichier par fichier. C'est la voie la plus simple et la plus robuste quand le montage est lent :

```bash
python jeanzay_pretraining_96k/stage_pretraining_96k_for_jeanzay.py \
  --output-root /tmp/dolphin_pretraining_96k_stage_meta \
  --archive-path /home/pablo/ibens/jz_work_ioc/dolphin_pretraining_96k_stage.tar \
  --reset-output
```

Cette commande :
- ecrit les metadonnees locales sous `/tmp/dolphin_pretraining_96k_stage_meta/generated`
- produit un seul fichier `dolphin_pretraining_96k_stage.tar` sur le workspace JZ monte
- evite le goulot d'etranglement de milliers de copies individuelles

Si tu veux juste mesurer ce qu'il faudrait transférer sans copier les données :

```bash
python jeanzay_pretraining_96k/stage_pretraining_96k_for_jeanzay.py \
  --output-root /home/pablo/jeanzay_pretraining_96k_stage \
  --manifest-only \
  --reset-output
```

### 2. Transférer vers Jean Zay

Exemple simple :

```bash
rsync -av /home/pablo/jeanzay_pretraining_96k_stage/ \
  jeanzay:/gpfsscratch/rech/.../dolphin_pretraining_96k_stage/

rsync -av /home/pablo/Documents/DolphinWhistleExtractor/ \
  jeanzay:/gpfsscratch/rech/.../DolphinWhistleExtractor/
```

Si tu as choisi l'archive, il n'y a rien d'autre a transferer que le `.tar`.
Sur Jean Zay, extrais-le dans le dossier de staging final :

```bash
mkdir -p /lustre/fswork/projects/rech/ioc/commun/dolphin_pretraining_96k_stage
tar -xf /lustre/fswork/projects/rech/ioc/commun/dolphin_pretraining_96k_stage.tar \
  -C /lustre/fswork/projects/rech/ioc/commun/dolphin_pretraining_96k_stage
```

### 3. Lancer le rebuild sur Jean Zay

Le job pousse directement vers le Hub. Il n'a pas besoin de `--save-local`.

```bash
cd /gpfsscratch/rech/.../DolphinWhistleExtractor
sbatch --account=<ton_compte> jeanzay_pretraining_96k/rebuild_pretraining_96k.sbatch
```

Variables utiles au moment du `sbatch` :

```bash
sbatch \
  --account=<ton_compte> \
  --export=ALL,REPO_DIR=/gpfsscratch/rech/.../DolphinWhistleExtractor,STAGE_ROOT=/gpfsscratch/rech/.../dolphin_pretraining_96k_stage,TARGET_REPO=dolphinteam/DolphinWhistle-Pretraining-96k \
  jeanzay_pretraining_96k/rebuild_pretraining_96k.sbatch
```

Si tu as besoin d'activer un environnement Python :

```bash
sbatch \
  --account=<ton_compte> \
  --export=ALL,ENV_ACTIVATE=/path/to/env/bin/activate \
  jeanzay_pretraining_96k/rebuild_pretraining_96k.sbatch
```

## Notes pratiques

- `path_config.json` est généré avec des chemins relatifs, donc il reste portable tant que tu transfères tout le dossier de staging.
- Le job SLURM lit les données staged et utilise le rebuild existant du repo.
- Le wrapper JZ n'introduit pas une deuxième logique métier. Il se contente de rediriger les chemins.
- Le staging résout aussi les collisions localement pour embarquer les bons enregistrements source.
- Si tu ecris directement vers un montage distant, la copie fichier par fichier peut etre tres lente. L'option `--archive-path` est recommandee dans ce cas.
