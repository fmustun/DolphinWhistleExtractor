import copy
import gc
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – plots are saved, never displayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


# ── PATHS & HYPER-PARAMETERS ───────────────────────────────────────────────────
IMG_SIZE         = 224
BATCH_SIZE       = 16
NUM_EPOCHS       = 50
PATIENCE         = 10
LEARNING_RATE    = 1e-3
RANDOM_STATE     = 7
NUM_WORKERS      = 0
DATASET_CSV      = '/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/dataset.csv'
SPECTROGRAMS_DIR = '/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/spectrograms'
MODELS_DIR       = 'models/run3'
FIGS_DIR         = 'figs/run3'

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD  = (0.229, 0.224, 0.225)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')
    return torch.device('cpu')


seed_everything(RANDOM_STATE)
DEVICE = get_device()
PIN_MEMORY = DEVICE.type == 'cuda'

try:
    torch.set_float32_matmul_precision('high')
except (AttributeError, RuntimeError):
    pass

print(f'Using device: {DEVICE}')


# ── SESSION-AWARE K-FOLD ───────────────────────────────────────────────────────
def extract_session_id(recording_name: str) -> str:
    """Strip _channel_N so simultaneous multi-channel recordings share one ID."""
    return re.sub(r'_channel_\d+$', '', str(recording_name).strip())


def session_aware_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build k-fold splits where whole recording sessions stay together.

    Standard StratifiedKFold splits individual samples randomly, so clips from
    the same session can appear in both train and validation — the model then
    validates on audio it has effectively already heard (data leakage).

    Here, each session is assigned to exactly one fold using greedy bin-packing
    on per-session class counts. This keeps entire sessions together while
    still trying to balance whistles and noise across folds. Validation for
    fold k contains only sessions never seen during training.

    Returns a list of (train_indices, val_indices) numpy arrays.
    """
    rng = random.Random(random_state)
    session_stats = (
        df.groupby('_session')['label']
        .agg(total='size', positives='sum')
        .reset_index()
    )
    session_stats['negatives'] = session_stats['total'] - session_stats['positives']

    sessions = session_stats.to_dict('records')
    rng.shuffle(sessions)
    sessions.sort(key=lambda row: (-row['total'], -abs(row['positives'] - row['negatives'])))

    target_total = max(session_stats['total'].sum() / n_splits, 1.0)
    target_pos = max(session_stats['positives'].sum() / n_splits, 1.0)
    target_neg = max(session_stats['negatives'].sum() / n_splits, 1.0)

    fold_total = [0] * n_splits
    fold_pos = [0] * n_splits
    fold_neg = [0] * n_splits
    session_to_fold: dict[str, int] = {}

    for row in sessions:
        best_fold = None
        best_score = None

        for fold in range(n_splits):
            next_total = fold_total[fold] + row['total']
            next_pos = fold_pos[fold] + row['positives']
            next_neg = fold_neg[fold] + row['negatives']

            score = (
                max(next_total / target_total, next_pos / target_pos, next_neg / target_neg),
                abs(next_pos - target_pos) + abs(next_neg - target_neg),
                next_total,
                fold,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold

        assert best_fold is not None
        session_to_fold[row['_session']] = best_fold
        fold_total[best_fold] += int(row['total'])
        fold_pos[best_fold] += int(row['positives'])
        fold_neg[best_fold] += int(row['negatives'])

    print('  Session distribution → folds:')
    for i in range(n_splits):
        print(
            f'    fold{i + 1}: total={fold_total[i]}  '
            f'whistles={fold_pos[i]}  noise={fold_neg[i]}'
        )

    fold_assignments = df['_session'].map(session_to_fold)
    if fold_assignments.isna().any():
        missing = sorted(df.loc[fold_assignments.isna(), '_session'].unique())
        raise RuntimeError(f'Missing fold assignment for sessions: {missing[:5]}')
    fold_assignments = fold_assignments.astype(int)

    splits = []
    for fold in range(n_splits):
        val_idx = fold_assignments[fold_assignments == fold].index.to_numpy()
        train_idx = fold_assignments[fold_assignments != fold].index.to_numpy()
        train_sessions = set(df.loc[train_idx, '_session'])
        val_sessions = set(df.loc[val_idx, '_session'])
        overlap = train_sessions & val_sessions
        if overlap:
            raise RuntimeError(
                f'Session leakage detected in fold {fold + 1}: '
                f'{sorted(overlap)[:5]}'
            )
        splits.append((train_idx, val_idx))

    return splits


# ── DATA ───────────────────────────────────────────────────────────────────────
raw = pd.read_csv(DATASET_CSV)

# Derive spectrogram path relative to SPECTROGRAMS_DIR.
# The WAV stem (e.g. "whistle_023193") matches the JPEG stem exactly.
raw['file_names'] = raw.apply(
    lambda r: os.path.join(
        'positives' if r['label'] == 1 else 'negatives',
        Path(r['file_path']).stem + '.jpg',
    ),
    axis=1,
)
raw['_session'] = raw['recording'].apply(extract_session_id)

n_pos = (raw['label'] == 1).sum()
n_neg = (raw['label'] == 0).sum()
print(f"Dataset: {len(raw)} samples  (whistles={n_pos}  noise={n_neg})")
print(f"Unique sessions: {raw['_session'].nunique()}\n")

print('Building session-aware 5-fold splits …')
splits = session_aware_kfold(raw, n_splits=5, random_state=RANDOM_STATE)

print('\nFold composition:')
for i, (tr_idx, va_idx) in enumerate(splits, 1):
    tr, va = raw.loc[tr_idx], raw.loc[va_idx]
    print(
        f"  Fold {i}: train={len(tr_idx):6d} ({tr['_session'].nunique()} sessions)  "
        f"val={len(va_idx):6d} ({va['_session'].nunique()} sessions)  "
        f"[val whistles={(va['label'] == 1).sum()}  val noise={(va['label'] == 0).sum()}]"
    )


# ── DATASETS & DATALOADERS ─────────────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        spectrograms_dir: str,
        transform: transforms.Compose,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.spectrograms_dir = Path(spectrograms_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = self.spectrograms_dir / row['file_names']

        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = self.transform(image)

        label = torch.tensor(int(row['label']), dtype=torch.long)
        return image, label


# For spectrograms (x=time, y=frequency):
#   horizontal_flip  → time reversal, acceptable for whistle shapes
#   width_shift      → small time translation, acceptable
#   brightness_range → simulates different recording conditions
#   height_shift / vertical_flip intentionally omitted: would shift/flip the
#   frequency axis, producing physically meaningless inputs.
common_transforms = [
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
    # Torchvision pretrained VGG16 expects ImageNet normalization.
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
]
train_transform = transforms.Compose(common_transforms)
valid_transform = transforms.Compose(common_transforms)


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────
def load_pretrained_vgg16() -> models.VGG:
    try:
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print(f'Warning: unable to load pretrained VGG16 weights ({exc}).')
        print('Falling back to randomly initialized VGG16.')
        return models.vgg16(weights=None)


class VGG16WhistleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = load_pretrained_vgg16()
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Freeze the entire VGG16 base – only the Dense head is trained.
        # Fine-tuning the full network on this spectrogram dataset caused
        # extreme overfitting in the original TensorFlow pipeline.
        for param in self.features.parameters():
            param.requires_grad = False

        in_features = backbone.classifier[0].in_features
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 50)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc3(x))
        return self.out(x)


def build_model() -> VGG16WhistleModel:
    return VGG16WhistleModel().to(DEVICE)


def build_optimizer(model: VGG16WhistleModel) -> Adam:
    return Adam(
        [
            {'params': [model.fc1.weight], 'weight_decay': 1e-4},
            {
                'params': [
                    model.fc1.bias,
                    model.fc2.weight,
                    model.fc2.bias,
                    model.fc3.weight,
                    model.fc3.bias,
                    model.out.weight,
                    model.out.bias,
                ],
                'weight_decay': 0.0,
            },
        ],
        lr=LEARNING_RATE,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None = None,
) -> dict[str, np.ndarray | float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_scores = []
    all_labels = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
            all_scores.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'scores': scores,
        'labels': labels,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'img_size': IMG_SIZE,
            'class_to_idx': {'0': 0, '1': 1},
            'normalization': {'mean': IMAGE_MEAN, 'std': IMAGE_STD},
            'architecture': 'torchvision_vgg16_frozen_features_custom_head',
        },
        path,
    )


def plot_training_curves(history: dict[str, list[float]], fold: int) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history['loss'], label='train')
    ax_loss.plot(history['val_loss'], label='val')
    ax_loss.set_title(f'Model loss – fold {fold}')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(history['accuracy'], label='train')
    ax_acc.plot(history['val_accuracy'], label='val')
    ax_acc.set_title(f'Model accuracy – fold {fold}')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'metrics_fold{fold}.png'))
    plt.close()


# ── CROSS-VALIDATION LOOP ──────────────────────────────────────────────────────
fpr_plot_list = []
tpr_plot_list = []
label_plot_list = []
list_loss = []
list_acc = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)

best_val_loss = np.inf
best_model_path = os.path.join(MODELS_DIR, 'model_vgg_best.pt')

criterion = nn.CrossEntropyLoss()

for fold, (train_idx, val_idx) in enumerate(splits, start=1):
    print(f'\n── Fold {fold}/5 ──────────────────────────────')

    training_data = raw.loc[train_idx].reset_index(drop=True)
    validation_data = raw.loc[val_idx].reset_index(drop=True)

    train_dataset = SpectrogramDataset(training_data, SPECTROGRAMS_DIR, train_transform)
    valid_dataset = SpectrogramDataset(validation_data, SPECTROGRAMS_DIR, valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    fold_ckpt_path = os.path.join(MODELS_DIR, f'model_vgg_fold{fold}.pt')

    net = build_model()
    optimizer = build_optimizer(net)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    best_fold_state = None
    best_fold_loss = np.inf
    best_fold_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(net, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch(net, valid_loader, criterion)
        scheduler.step(float(val_metrics['loss']))

        history['loss'].append(float(train_metrics['loss']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['accuracy'].append(float(train_metrics['accuracy']))
        history['val_accuracy'].append(float(val_metrics['accuracy']))

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"lr={current_lr:.2e}"
        )

        if float(val_metrics['loss']) < best_fold_loss:
            best_fold_loss = float(val_metrics['loss'])
            best_fold_epoch = epoch
            best_fold_state = copy.deepcopy(net.state_dict())
            save_checkpoint(fold_ckpt_path, net, epoch, best_fold_loss)
            print(f'  → Fold checkpoint saved  (val_loss = {best_fold_loss:.4f})')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f'  → Early stopping triggered after {epoch} epochs.')
            break

    if best_fold_state is not None:
        net.load_state_dict(best_fold_state)

    plot_training_curves(history, fold)

    eval_metrics = run_epoch(net, valid_loader, criterion)
    loss = float(eval_metrics['loss'])
    acc = float(eval_metrics['accuracy'])
    list_loss.append(loss)
    list_acc.append(acc)

    # ── ROC ───────────────────────────────────────────────────────────────────
    if np.unique(eval_metrics['labels']).size >= 2:
        fpr, tpr, _ = roc_curve(eval_metrics['labels'], eval_metrics['scores'], pos_label=1)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)

        fpr_plot_list.append(fpr)
        tpr_plot_list.append(tpr)
        label_plot_list.append(f'ROC fold {fold} (AUC = {roc_auc:.2f})')
    else:
        print('  → ROC skipped: validation fold contains a single class.')

    if loss < best_val_loss:
        best_val_loss = loss
        save_checkpoint(best_model_path, net, best_fold_epoch, best_val_loss)
        print(f'  → New best model saved  (val_loss = {best_val_loss:.4f})')

    del training_data, validation_data, train_dataset, valid_dataset
    del train_loader, valid_loader, net, optimizer, scheduler, best_fold_state
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f'\nAvg loss: {np.mean(list_loss):.4f}  |  Avg accuracy: {np.mean(list_acc):.4f}')
print(f'Best model saved to: {best_model_path}  (val_loss = {best_val_loss:.4f})')


# ── MEAN ROC ───────────────────────────────────────────────────────────────────
if tprs:
    plt.figure(figsize=(8, 6))
    for i in range(len(tpr_plot_list)):
        plt.plot(
            fpr_plot_list[i],
            tpr_plot_list[i],
            lw=2,
            alpha=0.3,
            label=label_plot_list[i],
        )

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – 5-fold cross-validation')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'avg_roc_vgg.png'))
    plt.close()
else:
    print('No ROC curve generated because no validation fold contained both classes.')
