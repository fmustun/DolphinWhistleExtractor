import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – plots are saved, never displayed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
import numpy as np
import gc

# ── PATHS & HYPER-PARAMETERS ───────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
DIRECTORY  = "DNN_whistle_detection/Train_model"
MODELS_DIR = "models"
FIGS_DIR   = "figs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)

# ── DATA ───────────────────────────────────────────────────────────────────────
train_data = pd.read_csv('Train_model/new_dataset.csv')
train_data = train_data.sample(frac=1, random_state=2).reset_index(drop=True)
train_data = train_data.astype({"labels": str})

labels = train_data['labels']

skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

# Augmentation only for training. For spectrograms (x=time, y=frequency):
#   horizontal_flip  → time reversal, acceptable for whistle shapes
#   width_shift      → small time translation, acceptable
#   brightness_range → simulates different recording conditions
#   height_shift / vertical_flip are intentionally omitted: they would shift or
#   invert the frequency axis, producing physically meaningless inputs.
train_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    width_shift_range=0.05,
    brightness_range=[0.85, 1.15],
)
valid_idg = ImageDataGenerator(preprocessing_function=preprocess_input)


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────
def build_model(img_size: int) -> models.Sequential:
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(img_size, img_size, 3))
    # Phase 1: freeze the entire VGG16 base.
    # Fine-tuning all 138 M parameters on a small spectrogram dataset causes
    # extreme overfitting (train loss → 1e-9 while inference fails completely).
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


def unfreeze_top_block(net: models.Sequential) -> None:
    """Unfreeze VGG16 block5 only – used for Phase 2 fine-tuning."""
    vgg = net.layers[0]  # first layer of Sequential is the VGG16 sub-model
    vgg.trainable = True
    for layer in vgg.layers:
        layer.trainable = layer.name.startswith('block5')


# ── CROSS-VALIDATION LOOP ──────────────────────────────────────────────────────
fpr_plot_list   = []
tpr_plot_list   = []
label_plot_list = []
list_loss       = []
list_acc        = []
tprs            = []
mean_fpr        = np.linspace(0, 1, 100)

best_val_loss   = np.inf
best_model_path = os.path.join(MODELS_DIR, "model_vgg_best.h5")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, labels), start=1):
    print(f"\n── Fold {fold}/5 ──────────────────────────────")

    training_data   = train_data.iloc[train_idx]
    validation_data = train_data.iloc[val_idx]
    del train_idx, val_idx
    gc.collect()

    # Each fold starts from a fresh model so validation metrics are independent
    # and represent true generalisation on held-out data.
    train_gen = train_idg.flow_from_dataframe(
        dataframe=training_data,
        directory=DIRECTORY,
        x_col='file_names',
        y_col='labels',
        batch_size=BATCH_SIZE,
        seed=None,
        shuffle=True,
        class_mode='categorical',
        classes=["0", "1"],
        color_mode="rgb",
        target_size=(IMG_SIZE, IMG_SIZE),
    )
    valid_gen = valid_idg.flow_from_dataframe(
        dataframe=validation_data,
        directory=DIRECTORY,
        x_col='file_names',
        y_col='labels',
        batch_size=BATCH_SIZE,
        seed=None,
        shuffle=False,
        class_mode='categorical',
        classes=["0", "1"],
        color_mode="rgb",
        target_size=(IMG_SIZE, IMG_SIZE),
    )

    fold_ckpt_path = os.path.join(MODELS_DIR, f"model_vgg_fold{fold}.h5")

    # ── Phase 1: train Dense head only (VGG16 frozen) ────────────────────────
    print(f"  Phase 1 – training Dense head, VGG16 frozen …")
    net = build_model(IMG_SIZE)
    net.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    phase1_callbacks = [
        ModelCheckpoint(filepath=fold_ckpt_path, monitor='val_loss',
                        save_best_only=True, mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=10,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
    ]
    history1 = net.fit(
        train_gen,
        epochs=50,
        validation_data=valid_gen,
        callbacks=phase1_callbacks,
    )

    # ── Phase 2: fine-tune VGG16 block5 at a much smaller learning rate ──────
    print(f"  Phase 2 – fine-tuning VGG16 block5 …")
    unfreeze_top_block(net)
    net.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    phase2_callbacks = [
        ModelCheckpoint(filepath=fold_ckpt_path, monitor='val_loss',
                        save_best_only=True, mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', mode='min', patience=10,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                          min_lr=1e-7, verbose=1),
    ]
    history2 = net.fit(
        train_gen,
        epochs=30,
        validation_data=valid_gen,
        callbacks=phase2_callbacks,
    )

    # ── Training curves (both phases concatenated) ────────────────────────────
    p1 = len(history1.history['loss'])
    combined = {
        k: history1.history[k] + history2.history[k]
        for k in ('loss', 'val_loss', 'accuracy', 'val_accuracy')
    }

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(combined['loss'],     label='train')
    ax_loss.plot(combined['val_loss'], label='val')
    ax_loss.axvline(p1, color='gray', linestyle='--', alpha=0.7,
                    label='phase 2 start')
    ax_loss.set_title(f"Model loss – fold {fold}")
    ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss')
    ax_loss.legend(loc='upper right')

    ax_acc.plot(combined['accuracy'],     label='train')
    ax_acc.plot(combined['val_accuracy'], label='val')
    ax_acc.axvline(p1, color='gray', linestyle='--', alpha=0.7,
                   label='phase 2 start')
    ax_acc.set_title(f"Model accuracy – fold {fold}")
    ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f"metrics_fold{fold}.png"))
    plt.close()

    # ── Evaluation ────────────────────────────────────────────────────────────
    loss, acc = net.evaluate(valid_gen, verbose=2)
    list_loss.append(loss)
    list_acc.append(acc)

    # ── ROC ───────────────────────────────────────────────────────────────────
    valid_gen.reset()
    predictions = net.predict(valid_gen)
    fpr, tpr, _ = roc_curve(
        validation_data["labels"].astype(int), predictions[:, 1], pos_label=1
    )
    interp_tpr    = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    roc_auc = auc(fpr, tpr)

    fpr_plot_list.append(fpr)
    tpr_plot_list.append(tpr)
    label_plot_list.append(f"ROC fold {fold} (AUC = {roc_auc:.2f})")

    # ── Track globally best model across all folds ─────────────────────────────
    if loss < best_val_loss:
        best_val_loss = loss
        net.save(best_model_path)
        print(f"  → New best model saved  (val_loss = {best_val_loss:.4f})")

    del training_data, validation_data, net
    gc.collect()

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f"\nAvg loss: {np.mean(list_loss):.4f}  |  Avg accuracy: {np.mean(list_acc):.4f}")
print(f"Best model saved to: {best_model_path}  (val_loss = {best_val_loss:.4f})")

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
         label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC – 5-fold cross-validation')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "avg_roc_vgg.png"))
plt.close()
