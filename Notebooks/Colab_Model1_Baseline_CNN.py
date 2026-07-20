# ================================================================================
# Model1_Baseline_CNN.ipynb — Group 15 | 7PAM2033
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use("seaborn-v0_8-whitegrid")
logger.info("TensorFlow version: %s", tf.__version__)


# ── CELL 2: Paths and settings ───────────────────────────────────────────────────
# All paths point to Google Drive — no changes needed

DATA_DIR    = Path("/content/drive/MyDrive/MRI_Sample")    # 1000 images per class
RESULTS_DIR = Path("/content/drive/MyDrive/Group15_Results")
MODELS_DIR  = Path("/content/drive/MyDrive/Group15_Results/models")
PLOTS_DIR   = Path("/content/drive/MyDrive/Group15_Results/plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES    = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
N_CLASSES  = len(CLASSES)
IMG_SIZE   = (224, 224)
IMG_SHAPE  = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS     = 50
VAL_SPLIT  = 0.15
LEARN_RATE = 0.001

logger.info("DATA_DIR = %s", DATA_DIR.resolve())


# ── CELL 3: Validate dataset ─────────────────────────────────────────────────────
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"MRI_Sample not found at {DATA_DIR}\n"
        "Please run the sample creation cell first."
    )

print("Images per class:")
total = 0
for cls in CLASSES:
    n = len(list((DATA_DIR / cls).glob("*.jpg")) +
            list((DATA_DIR / cls).glob("*.png")))
    print(f"  {cls:<22}: {n}")
    total += n
print(f"\n  Total: {total:,}")


# ── CELL 4: Data generators ──────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0/255, rotation_range=10,
    width_shift_range=0.05, height_shift_range=0.05,
    horizontal_flip=True, zoom_range=0.05,
    validation_split=VAL_SPLIT
)
eval_datagen = ImageDataGenerator(rescale=1.0/255)

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

print(f"\n  Train : {train_gen.samples:,}")
print(f"  Val   : {val_gen.samples:,}")
print(f"  Test  : {test_gen.samples:,}")


# ── CELL 5: Build Baseline CNN ───────────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32,  (3,3), padding="same", activation="relu", input_shape=IMG_SHAPE),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.25),

    layers.Conv2D(64,  (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(), layers.Dropout(0.5),
    layers.Dense(N_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE),
    loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()


# ── CELL 6: Callbacks ────────────────────────────────────────────────────────────
training_callbacks = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint(
        str(MODELS_DIR / "model1_baseline_cnn_best.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-7, verbose=1)
]


# ── CELL 7: Train ────────────────────────────────────────────────────────────────
print("Training Model 1 — Baseline CNN...")
history = model.fit(
    train_gen, epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=training_callbacks, verbose=1
)
print(f"\nBest val accuracy: {max(history.history['val_accuracy']):.4f}")


# ── CELL 8: Training curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train", color="#2196F3", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val",   color="#FF9800", linewidth=2, linestyle="--")
axes[0].set_title("Accuracy", fontweight="bold"); axes[0].legend(); axes[0].set_ylim(0, 1)
axes[1].plot(history.history["loss"],     label="Train", color="#4CAF50", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val",   color="#F44336", linewidth=2, linestyle="--")
axes[1].set_title("Loss", fontweight="bold"); axes[1].legend()
plt.suptitle("Baseline CNN — Training History", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model1_training_history.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 9: Evaluate ─────────────────────────────────────────────────────────────
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = test_gen.classes

accuracy = accuracy_score(y_true, y_pred)
f1       = f1_score(y_true, y_pred, average="weighted")
y_true_oh = tf.keras.utils.to_categorical(y_true, N_CLASSES)
roc_auc   = roc_auc_score(y_true_oh, y_pred_probs, multi_class="ovr", average="weighted")

print(f"\n  Accuracy : {accuracy:.4f}  {'✅' if accuracy >= 0.85 else '❌'} (KPI: ≥ 0.85)")
print(f"  F1 Score : {f1:.4f}  {'✅' if f1 >= 0.85 else '❌'} (KPI: ≥ 0.85)")
print(f"  ROC-AUC  : {roc_auc:.4f}  {'✅' if roc_auc >= 0.90 else '❌'} (KPI: ≥ 0.90)")
print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))


# ── CELL 10: Confusion matrix ────────────────────────────────────────────────────
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm,      annot=True, fmt="d",    cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
sns.heatmap(cm_norm, annot=True, fmt=".2f",  cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1], vmin=0, vmax=1)
axes[0].set_title("Confusion Matrix (Counts)",     fontweight="bold"); axes[0].tick_params(axis="x", rotation=15)
axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold"); axes[1].tick_params(axis="x", rotation=15)
plt.suptitle("Baseline CNN — Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model1_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 11: Save predictions and model ──────────────────────────────────────────
pred_df = pd.DataFrame(y_pred_probs, columns=[f"prob_{c}" for c in CLASSES])
pred_df["predicted_class"] = y_pred
pred_df["predicted_label"] = [CLASSES[i] for i in y_pred]
pred_df["true_class"]      = y_true
pred_df["true_label"]      = [CLASSES[i] for i in y_true]
pred_df["correct"]         = (y_pred == y_true)
pred_df.to_csv(RESULTS_DIR / "model1_predictions.csv", index=False)

model.save(str(MODELS_DIR / "model1_baseline_cnn_final.h5"))

print(f"\nPredictions saved: Group15_Results/model1_predictions.csv")
print(f"Model saved      : Group15_Results/models/model1_baseline_cnn_final.h5")
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nNext → Model2_TransferLearning.ipynb")
