# ================================================================================
# Model2_TransferLearning.ipynb — Group 15 | 7PAM2033
# Colab Version — paths already set for Google Drive
# ================================================================================


# ── CELL 1: Imports ──────────────────────────────────────────────────────────────
import os, random, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
plt.style.use("seaborn-v0_8-whitegrid")
logger.info("TensorFlow version: %s", tf.__version__)


# ── CELL 2: Paths and settings ───────────────────────────────────────────────────
# All paths point to Google Drive — no changes needed

DATA_DIR    = Path("/content/drive/MyDrive/MRI_Sample")
RESULTS_DIR = Path("/content/drive/MyDrive/Group15_Results")
MODELS_DIR  = Path("/content/drive/MyDrive/Group15_Results/models")
PLOTS_DIR   = Path("/content/drive/MyDrive/Group15_Results/plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES          = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
N_CLASSES        = len(CLASSES)
IMG_SIZE         = (224, 224)
IMG_SHAPE        = (224, 224, 3)
BATCH_SIZE       = 32
EPOCHS_FROZEN    = 15
EPOCHS_FINETUNE  = 20
VAL_SPLIT        = 0.15
LEARN_RATE_HEAD  = 0.001
LEARN_RATE_FINE  = 0.0001


# ── CELL 3: Data generators ──────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10, width_shift_range=0.05,
    height_shift_range=0.05, horizontal_flip=True,
    zoom_range=0.05, validation_split=VAL_SPLIT
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    str(DATA_DIR), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES,
    subset="training", shuffle=True, seed=SEED
)
val_gen = train_datagen.flow_from_directory(
    str(DATA_DIR), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES,
    subset="validation", shuffle=False, seed=SEED
)
test_gen = eval_datagen.flow_from_directory(
    str(DATA_DIR), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=CLASSES, shuffle=False
)
print(f"Train: {train_gen.samples:,} | Val: {val_gen.samples:,} | Test: {test_gen.samples:,}")


# ── CELL 4: Build EfficientNetB4 model ───────────────────────────────────────────
base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)
base_model.trainable = False

inputs  = layers.Input(shape=IMG_SHAPE)
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
model   = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_HEAD),
    loss="categorical_crossentropy", metrics=["accuracy"]
)
print(f"Phase 1 model built — {model.count_params():,} parameters")


# ── CELL 5: Phase 1 training — head only ─────────────────────────────────────────
phase1_callbacks = [
    callbacks.EarlyStopping(monitor="val_loss", patience=7,
                            restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / "model2_efficientnet_phase1.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-7, verbose=1)
]

print("Phase 1: Training head (base frozen)...")
history_phase1 = model.fit(
    train_gen, epochs=EPOCHS_FROZEN,
    validation_data=val_gen,
    callbacks=phase1_callbacks, verbose=1
)
print(f"Phase 1 best val accuracy: {max(history_phase1.history['val_accuracy']):.4f}")


# ── CELL 6: Phase 2 — fine-tune top 30 layers ────────────────────────────────────
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE_FINE),
    loss="categorical_crossentropy", metrics=["accuracy"]
)

phase2_callbacks = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / "model2_efficientnet_best.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-8, verbose=1)
]

print("Phase 2: Fine-tuning top 30 layers...")
history_phase2 = model.fit(
    train_gen, epochs=EPOCHS_FINETUNE,
    validation_data=val_gen,
    callbacks=phase2_callbacks, verbose=1
)
print(f"Phase 2 best val accuracy: {max(history_phase2.history['val_accuracy']):.4f}")


# ── CELL 7: Plot training history ────────────────────────────────────────────────
all_acc      = history_phase1.history["accuracy"]     + history_phase2.history["accuracy"]
all_val_acc  = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
all_loss     = history_phase1.history["loss"]         + history_phase2.history["loss"]
all_val_loss = history_phase1.history["val_loss"]     + history_phase2.history["val_loss"]
phase1_end   = len(history_phase1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(all_acc,     label="Train", color="#2196F3", linewidth=2)
axes[0].plot(all_val_acc, label="Val",   color="#FF9800", linewidth=2, linestyle="--")
axes[0].axvline(phase1_end, color="grey", linestyle=":", linewidth=1.5, label="Fine-tune starts")
axes[0].set_title("Accuracy", fontweight="bold"); axes[0].legend(); axes[0].set_ylim(0, 1)
axes[1].plot(all_loss,     label="Train", color="#4CAF50", linewidth=2)
axes[1].plot(all_val_loss, label="Val",   color="#F44336", linewidth=2, linestyle="--")
axes[1].axvline(phase1_end, color="grey", linestyle=":", linewidth=1.5, label="Fine-tune starts")
axes[1].set_title("Loss", fontweight="bold"); axes[1].legend()
plt.suptitle("EfficientNetB4 — Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model2_training_history.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 8: Evaluate ─────────────────────────────────────────────────────────────
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes

accuracy  = accuracy_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred, average="weighted")
y_true_oh = tf.keras.utils.to_categorical(y_true, N_CLASSES)
roc_auc   = roc_auc_score(y_true_oh, y_pred_probs, multi_class="ovr", average="weighted")

print(f"\n  Accuracy : {accuracy:.4f}  {'✅' if accuracy >= 0.85 else '❌'} (KPI: ≥ 0.85)")
print(f"  F1 Score : {f1:.4f}  {'✅' if f1 >= 0.85 else '❌'} (KPI: ≥ 0.85)")
print(f"  ROC-AUC  : {roc_auc:.4f}  {'✅' if roc_auc >= 0.90 else '❌'} (KPI: ≥ 0.90)")
print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))


# ── CELL 9: Confusion matrix ─────────────────────────────────────────────────────
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm,      annot=True, fmt="d",   cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], vmin=0, vmax=1)
for ax in axes:
    ax.tick_params(axis="x", rotation=15)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
axes[0].set_title("Counts", fontweight="bold")
axes[1].set_title("Normalised", fontweight="bold")
plt.suptitle("EfficientNetB4 — Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 10: Save predictions and model ──────────────────────────────────────────
pred_df = pd.DataFrame(y_pred_probs, columns=[f"prob_{c}" for c in CLASSES])
pred_df["predicted_class"] = y_pred
pred_df["predicted_label"] = [CLASSES[i] for i in y_pred]
pred_df["true_class"]      = y_true
pred_df["true_label"]      = [CLASSES[i] for i in y_true]
pred_df["correct"]         = (y_pred == y_true)
pred_df.to_csv(RESULTS_DIR / "model2_predictions.csv", index=False)
model.save(str(MODELS_DIR / "model2_efficientnet_final.h5"))

# Compare with Model 1
m1_path = RESULTS_DIR / "model1_predictions.csv"
if m1_path.exists():
    m1 = pd.read_csv(m1_path)
    m1_acc = accuracy_score(m1["true_class"], m1["predicted_class"])
    print(f"\n  Model 1 Accuracy : {m1_acc:.4f}")
    print(f"  Model 2 Accuracy : {accuracy:.4f}")
    print(f"  Improvement      : {(accuracy - m1_acc)*100:+.2f}%")

print(f"\nPredictions saved: Group15_Results/model2_predictions.csv")
print(f"Model saved      : Group15_Results/models/model2_efficientnet_final.h5")
print("\nNext → Model3_Hybrid_Multimodal.ipynb")
