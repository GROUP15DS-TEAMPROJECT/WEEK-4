# ================================================================================
# Model3_Hybrid_Multimodal.ipynb — Group 15 | 7PAM2033
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
plt.style.use("seaborn-v0_8-whitegrid")


# ── CELL 2: Paths and settings ───────────────────────────────────────────────────
# All paths point to Google Drive — no changes needed

MRI_DIR      = Path("/content/drive/MyDrive/MRI_Sample")
CLINICAL_CSV = Path("/content/drive/MyDrive/Clinical_Data.csv")
RESULTS_DIR  = Path("/content/drive/MyDrive/Group15_Results")
MODELS_DIR   = Path("/content/drive/MyDrive/Group15_Results/models")
PLOTS_DIR    = Path("/content/drive/MyDrive/Group15_Results/plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MRI_CLASSES    = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
N_CLASSES      = len(MRI_CLASSES)
IMG_SIZE       = (224, 224)
IMG_SHAPE      = (224, 224, 3)
BATCH_SIZE     = 32
DROP_COLS      = ["PatientID", "DoctorInCharge"]
TARGET_COL     = "Diagnosis"
SENSITIVE_COLS = ["Age", "Gender", "Ethnicity", "EducationLevel"]
MRI_WEIGHT     = 0.5
CLINICAL_WEIGHT = 0.5


# ── CELL 3: Load MRI model from Model 2 ──────────────────────────────────────────
# Reuse EfficientNetB4 from Model 2 as the MRI branch
mri_model_path = MODELS_DIR / "model2_efficientnet_best.h5"
if not mri_model_path.exists():
    mri_model_path = MODELS_DIR / "model1_baseline_cnn_best.h5"
    logger.warning("Model 2 not found — using Model 1 as MRI branch.")

mri_model = tf.keras.models.load_model(str(mri_model_path))
print(f"MRI branch loaded: {mri_model_path.name}")


# ── CELL 4: Get MRI predictions ──────────────────────────────────────────────────
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
mri_gen = eval_datagen.flow_from_directory(
    str(MRI_DIR), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", classes=MRI_CLASSES, shuffle=False
)

mri_gen.reset()
mri_probs = mri_model.predict(mri_gen, verbose=1)
mri_preds = np.argmax(mri_probs, axis=1)
mri_true  = mri_gen.classes

print(f"MRI branch accuracy: {accuracy_score(mri_true, mri_preds):.4f}")


# ── CELL 5: Load and preprocess clinical data ─────────────────────────────────────
df_clinical = pd.read_csv(CLINICAL_CSV)
drop = [c for c in DROP_COLS if c in df_clinical.columns]
df_clinical.drop(columns=drop, inplace=True)

sensitive_df = df_clinical[[c for c in SENSITIVE_COLS
                             if c in df_clinical.columns]].copy()

encoders = {}
for col in df_clinical.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_clinical[col] = le.fit_transform(df_clinical[col].astype(str))
    encoders[col] = le

X_clin = df_clinical.drop(columns=[TARGET_COL])
y_clin = df_clinical[TARGET_COL]

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X_clin, y_clin, sensitive_df,
    test_size=0.20, random_state=SEED, stratify=y_clin
)

scaler  = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_clin.columns)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_clin.columns)

print(f"Clinical data: Train={X_train.shape} | Test={X_test.shape}")


# ── CELL 6: Train XGBoost clinical branch ────────────────────────────────────────
classes_arr    = np.unique(y_train)
weights        = compute_class_weight("balanced", classes=classes_arr, y=y_train)
weight_dict    = dict(zip(classes_arr, weights))
sample_weights = y_train.map(weight_dict)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=SEED, n_jobs=-1
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights,
              eval_set=[(X_test, y_test)], verbose=50)

clin_probs = xgb_model.predict_proba(X_test)
clin_preds = np.argmax(clin_probs, axis=1)
print(f"\nClinical branch accuracy: {accuracy_score(y_test, clin_preds):.4f}")
print(classification_report(y_test, clin_preds,
                            target_names=["No Alzheimer's", "Alzheimer's"]))


# ── CELL 7: Feature importance ───────────────────────────────────────────────────
feat_importance = pd.Series(
    xgb_model.feature_importances_, index=X_clin.columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
feat_importance.head(20).plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 20 Clinical Feature Importances", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nTop 10 features:", feat_importance.head(10).to_string())


# ── CELL 8: Fusion — combine MRI + Clinical predictions ──────────────────────────
def expand_clinical_probs(clin_probs_2class):
    n        = len(clin_probs_2class)
    expanded = np.zeros((n, 4))
    expanded[:, 0] = clin_probs_2class[:, 0]
    expanded[:, 1] = clin_probs_2class[:, 1] * 0.40
    expanded[:, 2] = clin_probs_2class[:, 1] * 0.35
    expanded[:, 3] = clin_probs_2class[:, 1] * 0.25
    return expanded

clin_probs_4class = expand_clinical_probs(clin_probs)
n_test     = len(y_test)
sample_idx = np.random.choice(len(mri_probs), n_test, replace=False)
mri_sample = mri_probs[sample_idx]
mri_true_sample = mri_true[sample_idx]

hybrid_probs = MRI_WEIGHT * mri_sample + CLINICAL_WEIGHT * clin_probs_4class
hybrid_preds = np.argmax(hybrid_probs, axis=1)

hybrid_acc = accuracy_score(mri_true_sample, hybrid_preds)
hybrid_f1  = f1_score(mri_true_sample, hybrid_preds, average="weighted")
y_true_oh  = tf.keras.utils.to_categorical(mri_true_sample, N_CLASSES)
hybrid_auc = roc_auc_score(y_true_oh, hybrid_probs, multi_class="ovr", average="weighted")

print(f"\n  Hybrid Accuracy : {hybrid_acc:.4f}  {'✅' if hybrid_acc >= 0.85 else '❌'}")
print(f"  Hybrid F1       : {hybrid_f1:.4f}  {'✅' if hybrid_f1 >= 0.85 else '❌'}")
print(f"  Hybrid ROC-AUC  : {hybrid_auc:.4f}  {'✅' if hybrid_auc >= 0.90 else '❌'}")


# ── CELL 9: All models comparison ────────────────────────────────────────────────
results = {"Model 3 Hybrid": {"Accuracy": hybrid_acc, "F1": hybrid_f1, "ROC-AUC": hybrid_auc}}

for name, fname in [("Model 1 CNN", "model1_predictions.csv"),
                     ("Model 2 EffNet", "model2_predictions.csv")]:
    p = RESULTS_DIR / fname
    if p.exists():
        df = pd.read_csv(p)
        results[name] = {
            "Accuracy": accuracy_score(df["true_class"], df["predicted_class"]),
            "F1": f1_score(df["true_class"], df["predicted_class"], average="weighted"),
            "ROC-AUC": None
        }

df_results = pd.DataFrame(results).T
print("\n" + "="*50)
print("  ALL MODELS COMPARISON")
print("="*50)
print(df_results.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
df_results[["Accuracy", "F1"]].plot(kind="bar", ax=ax,
                                     color=["#2196F3", "#4CAF50"], edgecolor="white")
ax.axhline(0.85, color="red", linestyle="--", linewidth=1.5, label="KPI (0.85)")
ax.set_title("All Models — Accuracy and F1", fontweight="bold")
ax.set_ylim(0, 1); ax.legend(); ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model3_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# ── CELL 10: Save predictions and models ─────────────────────────────────────────
hybrid_df = pd.DataFrame(hybrid_probs, columns=[f"prob_{c}" for c in MRI_CLASSES])
hybrid_df["predicted_class"] = hybrid_preds
hybrid_df["predicted_label"] = [MRI_CLASSES[i] for i in hybrid_preds]
hybrid_df["true_class"]      = mri_true_sample
hybrid_df["true_label"]      = [MRI_CLASSES[i] for i in mri_true_sample]
hybrid_df["correct"]         = (hybrid_preds == mri_true_sample)
for col in SENSITIVE_COLS:
    if col in s_test.columns:
        hybrid_df[f"sensitive_{col}"] = s_test[col].values[:n_test]

hybrid_df.to_csv(RESULTS_DIR / "model3_predictions.csv", index=False)
joblib.dump(xgb_model, str(MODELS_DIR / "model3_xgboost_clinical.pkl"))
joblib.dump(scaler,    str(MODELS_DIR / "model3_clinical_scaler.pkl"))

print(f"Hybrid predictions saved : Group15_Results/model3_predictions.csv")
print(f"XGBoost model saved      : Group15_Results/models/model3_xgboost_clinical.pkl")
print("\nNext → Fairness_Evaluation.ipynb")
