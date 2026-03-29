import os
import re
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – plots are saved, never displayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import gc

# ── PATHS & HYPER-PARAMETERS ───────────────────────────────────────────────────
IMG_SIZE         = 224
BATCH_SIZE       = 16
DATASET_CSV      = '/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/dataset.csv'
SPECTROGRAMS_DIR = '/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset/spectrograms'
MODELS_DIR       = 'models/run3'
FIGS_DIR         = 'figs/run3'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)


# ── SESSION-AWARE K-FOLD ───────────────────────────────────────────────────────

def extract_session_id(recording_name: str) -> str:
    """Strip _channel_N so simultaneous multi-channel recordings share one ID."""
    return re.sub(r'_channel_\d+$', '', str(recording_name).strip())


def session_aware_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 7,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build k-fold splits where whole recording sessions stay together.

    Standard StratifiedKFold splits individual samples randomly, so clips from
    the same session can appear in both train and validation — the model then
    validates on audio it has effectively already heard (data leakage).

    Here, each session is assigned to exactly one fold using greedy bin-packing
    (largest session first), done separately per class to preserve class balance.
    Validation for fold k contains only sessions never seen during training.

    Returns a list of (train_indices, val_indices) numpy arrays.
    """
    rng = random.Random(random_state)
    fold_assignments = pd.Series(-1, index=df.index, dtype=int)

    for label_val, label_name in [(1, 'whistle'), (0, 'noise')]:
        label_mask = df['label'] == label_val
        label_sub  = df[label_mask]

        session_sizes = label_sub.groupby('_session').size()
        sessions = list(session_sizes.items())  # [(session_id, count), ...]

        # Shuffle to break ties randomly, then stable-sort largest-first for
        # better fold balance (greedy bin-packing heuristic)
        rng.shuffle(sessions)
        sessions.sort(key=lambda x: -x[1])

        fold_counts = [0] * n_splits
        for session_name, size in sessions:
            best = int(np.argmin(fold_counts))
            fold_counts[best] += size
            mask = label_mask & (df['_session'] == session_name)
            fold_assignments[mask] = best

        print(f"  {label_name.capitalize():8s} sessions → folds: "
              + "  ".join(f"fold{i+1}={c}" for i, c in enumerate(fold_counts)))

    splits = []
    for fold in range(n_splits):
        val_idx   = fold_assignments[fold_assignments == fold].index.to_numpy()
        train_idx = fold_assignments[fold_assignments != fold].index.to_numpy()
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
raw['labels']   = raw['label'].astype(str)
raw['_session'] = raw['recording'].apply(extract_session_id)

n_pos = (raw['label'] == 1).sum()
n_neg = (raw['label'] == 0).sum()
print(f"Dataset: {len(raw)} samples  (whistles={n_pos}  noise={n_neg})")
print(f"Unique sessions: {raw['_session'].nunique()}\n")

print("Building session-aware 5-fold splits …")
splits = session_aware_kfold(raw, n_splits=5, random_state=7)

print("\nFold composition:")
for i, (tr_idx, va_idx) in enumerate(splits, 1):
    tr, va = raw.loc[tr_idx], raw.loc[va_idx]
    print(f"  Fold {i}: train={len(tr_idx):6d} ({tr['_session'].nunique()} sessions)  "
          f"val={len(va_idx):6d} ({va['_session'].nunique()} sessions)  "
          f"[val whistles={(va['label']==1).sum()}  val noise={(va['label']==0).sum()}]")


# ── DATA GENERATORS ────────────────────────────────────────────────────────────
# For spectrograms (x=time, y=frequency):
#   horizontal_flip  → time reversal, acceptable for whistle shapes
#   width_shift      → small time translation, acceptable
#   brightness_range → simulates different recording conditions
#   height_shift / vertical_flip intentionally omitted: would shift/flip the
#   frequency axis, producing physically meaningless inputs.
train_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # horizontal_flip=True,
    # width_shift_range=0.05,
    # brightness_range=[0.85, 1.15],
)
valid_idg = ImageDataGenerator(preprocessing_function=preprocess_input)


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────
def build_model(img_size: int) -> models.Sequential:
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(img_size, img_size, 3))
    # Freeze the entire VGG16 base – only the Dense head is trained.
    # Fine-tuning 138 M parameters on a spectrogram dataset causes extreme
    # overfitting (train loss → 1e-9 while inference generalisation fails).
    base_model.trainable = False
    return models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(50,  activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(20,  activation='relu'),
        layers.Dense(2,   activation='softmax'),
    ], name='vgg16_whistle')


# ── CROSS-VALIDATION LOOP ──────────────────────────────────────────────────────
fpr_plot_list   = []
tpr_plot_list   = []
label_plot_list = []
list_loss       = []
list_acc        = []
tprs            = []
mean_fpr        = np.linspace(0, 1, 100)

best_val_loss   = np.inf
best_model_path = os.path.join(MODELS_DIR, 'model_vgg_best.h5')

for fold, (train_idx, val_idx) in enumerate(splits, start=1):
    print(f"\n── Fold {fold}/5 ──────────────────────────────")

    training_data   = raw.loc[train_idx].reset_index(drop=True)
    validation_data = raw.loc[val_idx].reset_index(drop=True)

    train_gen = train_idg.flow_from_dataframe(
        dataframe=training_data,
        directory=SPECTROGRAMS_DIR,
        x_col='file_names',
        y_col='labels',
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical',
        classes=['0', '1'],
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE),
    )
    valid_gen = valid_idg.flow_from_dataframe(
        dataframe=validation_data,
        directory=SPECTROGRAMS_DIR,
        x_col='file_names',
        y_col='labels',
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode='categorical',
        classes=['0', '1'],
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE),
    )

    fold_ckpt_path = os.path.join(MODELS_DIR, f'model_vgg_fold{fold}.h5')

    net = build_model(IMG_SIZE)
    net.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [
        ModelCheckpoint(filepath=fold_ckpt_path, monitor='val_loss',
                        save_best_only=True, mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=10,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]

    history = net.fit(
        train_gen,
        epochs=50,
        validation_data=valid_gen,
        callbacks=callbacks,
    )

    # ── Training curves ────────────────────────────────────────────────────────
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history.history['loss'],     label='train')
    ax_loss.plot(history.history['val_loss'], label='val')
    ax_loss.set_title(f'Model loss – fold {fold}')
    ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(history.history['accuracy'],     label='train')
    ax_acc.plot(history.history['val_accuracy'], label='val')
    ax_acc.set_title(f'Model accuracy – fold {fold}')
    ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'metrics_fold{fold}.png'))
    plt.close()

    # ── Evaluation ────────────────────────────────────────────────────────────
    loss, acc = net.evaluate(valid_gen, verbose=2)
    list_loss.append(loss)
    list_acc.append(acc)

    # ── ROC ───────────────────────────────────────────────────────────────────
    valid_gen.reset()
    predictions = net.predict(valid_gen)
    fpr, tpr, _ = roc_curve(
        validation_data['labels'].astype(int), predictions[:, 1], pos_label=1
    )
    interp_tpr    = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    roc_auc = auc(fpr, tpr)

    fpr_plot_list.append(fpr)
    tpr_plot_list.append(tpr)
    label_plot_list.append(f'ROC fold {fold} (AUC = {roc_auc:.2f})')

    if loss < best_val_loss:
        best_val_loss = loss
        net.save(best_model_path)
        print(f'  → New best model saved  (val_loss = {best_val_loss:.4f})')

    del training_data, validation_data, net
    gc.collect()

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f'\nAvg loss: {np.mean(list_loss):.4f}  |  Avg accuracy: {np.mean(list_acc):.4f}')
print(f'Best model saved to: {best_model_path}  (val_loss = {best_val_loss:.4f})')

# ── MEAN ROC ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
for i in range(len(tpr_plot_list)):
    plt.plot(fpr_plot_list[i], tpr_plot_list[i], lw=2, alpha=0.3,
             label=label_plot_list[i])

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr     = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc     = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC – 5-fold cross-validation')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'avg_roc_vgg.png'))
plt.close()
