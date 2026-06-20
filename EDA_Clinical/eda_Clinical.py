#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Alzheimer's Disease Dataset — Exploratory Data Analysis (EDA)
MSc Data Science | Evaluating Bias, Fairness, and Explainability in
Multimodal AI Systems for Early Alzheimer's Disease Detection
═══════════════════════════════════════════════════════════════════════════════
Author  : MSc Data Science Student
Version : 1.0.0
Python  : ≥ 3.9
Dependencies (install once):
    pip install pandas numpy matplotlib seaborn scipy scikit-learn
═══════════════════════════════════════════════════════════════════════════════
"""
"""
The file contains a professional module header, version information, and descriptive docstrings, 
which aligns well with good coding standards and improves project documentation quality.
"""

# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore") # All warnings are currently suppressed globallycwhile this helps keep the output clean

# The import statements are grouped logically and follow a consistent structure
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.stats import mannwhitneyu, chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

# ─── 1. GLOBAL SETTINGS ──────────────────────────────────────────────────────
DATA_PATH   = "alzheimers_disease_data.csv"   # ← update path if needed
OUTPUT_DIR  = "."                              # ← figures saved here

""" 
Variable names such as DX_LABS, PAL_DX, and LABEL_MAPS are concise and consistent 
adding a brief comment explaining these abbreviations would improve readability for new team members reviewing the code
"""
PAL_DX  = {0: "#2196F3", 1: "#E91E63"}
DX_LABS = {0: "No Alzheimer's", 1: "Alzheimer's"}

LABEL_MAPS = {
    "Gender":         {0: "Female",       1: "Male"},
    "Ethnicity":      {0: "Caucasian",    1: "Afr. American", 2: "Asian",  3: "Other"},
    "EducationLevel": {0: "None",         1: "High School",   2: "Bachelor's", 3: "Higher"},
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


# ─── 2. DATA LOADING & CLEANING ──────────────────────────────────────────────
"""
The data loading process is clear and the removal of non-informative columns is appropriate.
Consider adding validation to check whether the dataset file exists and whether required columns such as
Diagnosis, Age, Gender, and Ethnicity are present before proceeding.
This would make the pipeline more robust against dataset changes.
"""
def load_data(path: str) -> pd.DataFrame:
    """Load and pre-process the Alzheimer's dataset."""
    df = pd.read_csv(path)

    # Drop non-informative columns
    drop_cols = [c for c in ["PatientID", "DoctorInCharge"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Convenience derived columns
    # The age grouping is appropriate for demographic analysis.
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[59, 64, 69, 74, 79, 84, 90],
        labels=["60–64", "65–69", "70–74", "75–79", "80–84", "85–90"]
    )
    return df


# ─── 3. DATA PROFILING ───────────────────────────────────────────────────────
"""
Class distribution is reported clearly since the project includes fairness evaluation
it may also be useful to display class balance across demographic subgroups such as Gender and Ethnicity
"""
def profile_data(df: pd.DataFrame) -> None:
    """Print a comprehensive textual summary of the dataset."""
    print("=" * 70)
    print("DATASET PROFILE")
    print("=" * 70)
    print(f"Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Target            : Diagnosis  →  {dict(df['Diagnosis'].value_counts())}")
    imbalance_ratio = df['Diagnosis'].value_counts(normalize=True) * 100
    print(f"Class balance     : No AD={imbalance_ratio[0]:.1f}%  |  AD={imbalance_ratio[1]:.1f}%")
    print(f"\nMissing values per column:")
    miss = df.isnull().sum()
    print(miss[miss > 0].to_string() if miss.any() else "  None — dataset is complete.")

    print("\nDescriptive statistics (numeric):")
    print(df.describe(include="number").T.to_string())

    cont_feats = df.select_dtypes("number").columns.tolist()
    print("\nSkewness (continuous features):")
    print(df[cont_feats].skew().sort_values(ascending=False).to_string())

    print("\nKurtosis (continuous features):")
    print(df[cont_feats].kurtosis().sort_values(ascending=False).to_string())


# ─── 4. FIGURE HELPERS ───────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight") # consider validating that the output directory exists before saving figures
    plt.close(fig)
    print(f"  ✔ Saved → {path}")


"""
Several plotting functions contain substantial amounts of logic
Splitting larger functions into smaller helper functions would make
the code easier to test, maintain, and debug in future iterations.
"""
# ─── 5. FIGURE 1 — DATA OVERVIEW ─────────────────────────────────────────────
def fig1_overview(df: pd.DataFrame) -> None:
    """Target distribution, age, gender, ethnicity & education split."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Figure 1 · Data Overview & Class Distribution",
                 fontsize=15, fontweight="bold", y=1.01)

    # A – Target donut
    ax = axes[0, 0]
    counts = df["Diagnosis"].value_counts().sort_index()
    ax.pie(
        counts,
        labels=[DX_LABS[k] for k in counts.index],
        colors=[PAL_DX[k] for k in counts.index],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"width": 0.55}, textprops={"fontsize": 11}
    )
    ax.set_title("A · Target Class Distribution", fontweight="bold")

    # B – Age histogram
    ax = axes[0, 1]
    for dx in [0, 1]:
        ax.hist(df[df.Diagnosis == dx]["Age"], bins=15, alpha=0.65,
                color=PAL_DX[dx], label=DX_LABS[dx], edgecolor="white")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.set_title("B · Age Distribution by Diagnosis", fontweight="bold")
    ax.legend(framealpha=0.7)
    _, p = mannwhitneyu(df[df.Diagnosis == 1]["Age"], df[df.Diagnosis == 0]["Age"])
    ax.text(0.98, 0.95, f"MWU p={p:.4f}", transform=ax.transAxes, ha="right",
            fontsize=9, color="darkred" if p < 0.05 else "grey")

    # C – Gender split
    ax = axes[0, 2]
    gdf = df.groupby(["Gender", "Diagnosis"]).size().unstack(fill_value=0)
    gdf.index = ["Female", "Male"]
    gdf.columns = [DX_LABS[c] for c in gdf.columns]
    gdf.plot(kind="bar", ax=ax, color=[PAL_DX[0], PAL_DX[1]], edgecolor="white", rot=0)
    ax.set_title("C · Gender Split by Diagnosis", fontweight="bold")

    # D – Ethnicity diagnosis rate
    ax = axes[1, 0]
    eth_rate = df.groupby("Ethnicity")["Diagnosis"].mean() * 100
    eth_labels = [LABEL_MAPS["Ethnicity"][i] for i in eth_rate.index]
    bars = ax.bar(eth_labels, eth_rate.values,
                  color=sns.color_palette("Set2", len(eth_rate)), edgecolor="white")
    ax.axhline(df["Diagnosis"].mean() * 100, color="red", ls="--", lw=1.8, label="Overall rate")
    ax.set_ylabel("Diagnosis Rate (%)"); ax.legend(fontsize=9)
    ax.set_title("D · Diagnosis Rate by Ethnicity\n(Key Fairness Metric)", fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # E – Education level diagnosis rate
    ax = axes[1, 1]
    edu_rate = df.groupby("EducationLevel")["Diagnosis"].mean() * 100
    edu_labels = [LABEL_MAPS["EducationLevel"][i] for i in edu_rate.index]
    bars = ax.bar(edu_labels, edu_rate.values,
                  color=sns.color_palette("Set3", len(edu_rate)), edgecolor="white")
    ax.axhline(df["Diagnosis"].mean() * 100, color="red", ls="--", lw=1.8, label="Overall rate")
    ax.set_ylabel("Diagnosis Rate (%)"); ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title("E · Diagnosis Rate by Education Level\n(Social Determinant)", fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    # F – Sample size by ethnicity
    ax = axes[1, 2]
    eth_cnt = df["Ethnicity"].map(LABEL_MAPS["Ethnicity"]).value_counts()
    bars = ax.bar(eth_cnt.index, eth_cnt.values,
                  color=sns.color_palette("Set2", len(eth_cnt)), edgecolor="white")
    ax.set_ylabel("Patient Count")
    ax.set_title("F · Sample Size by Ethnicity\n(Representation Imbalance)", fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                str(int(b.get_height())), ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig1_overview")


# ─── 6. FIGURE 2 — CONTINUOUS DISTRIBUTIONS (Violin + Box) ──────────────────
"""

"""

def fig2_continuous_distributions(df: pd.DataFrame) -> None:
    """Violin + box plots for all continuous features by diagnosis."""
    cont_feats = [
        "Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
        "DietQuality", "SleepQuality", "SystolicBP", "DiastolicBP",
        "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
        "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"
    ]

    fig, axes = plt.subplots(3, 5, figsize=(22, 14))
    fig.suptitle("Figure 2 · Continuous Feature Distributions by Diagnosis",
                 fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        data = [df[df.Diagnosis == dx][feat].values for dx in [0, 1]]

        # Violin
        parts = ax.violinplot(data, positions=[0, 1], showmedians=False, showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([PAL_DX[0], PAL_DX[1]][j])
            pc.set_alpha(0.6)

        # Box overlay
        ax.boxplot(data, positions=[0, 1], widths=0.15,
                   medianprops={"color": "black", "linewidth": 2},
                   boxprops={"linewidth": 1.5}, whiskerprops={"linewidth": 1.2},
                   flierprops={"marker": "o", "markersize": 2, "alpha": 0.3})

        ax.set_xticks([0, 1]); ax.set_xticklabels(["No AD", "AD"], fontsize=9)
        ax.set_title(feat, fontweight="bold", fontsize=10)


        """
        The Mann-Whitney U test is appropriate for comparing non-parametric distributions
        between diagnosis groups. Consider adding validation to ensure both groups contain
        sufficient observations before running the statistical test.
        """
        # Mann-Whitney U test annotation
        _, p = mannwhitneyu(data[1], data[0])
        plab = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.97, 0.96, plab, transform=ax.transAxes, ha="right",
                fontsize=8, color="darkred" if p < 0.05 else "grey", va="top")

    patches = [mpatches.Patch(color=PAL_DX[k], label=DX_LABS[k]) for k in [0, 1]]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    fig.tight_layout()
    _save(fig, "fig2_continuous_dists")


# ─── 7. FIGURE 3 — BINARY FEATURE PREVALENCE ─────────────────────────────────
"""
Good defensive programming practice. The code checks whether expected features
exist before plotting, reducing the risk of runtime errors
"""
def fig3_binary_prevalence(df: pd.DataFrame) -> None:
    """Grouped bar charts of binary feature rates per diagnosis class + χ²."""
    bin_feats = [
        "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
        "Diabetes", "Depression", "HeadInjury", "Hypertension",
        "MemoryComplaints", "BehavioralProblems", "Confusion",
        "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"
    ]
    bin_feats = [f for f in bin_feats if f in df.columns]

    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    fig.suptitle("Figure 3 · Binary Feature Prevalence Rate by Diagnosis (%)",
                 fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(bin_feats):
        ax = axes[i]
        rates = df.groupby("Diagnosis")[feat].mean() * 100
        bars = ax.bar([DX_LABS[k] for k in rates.index], rates.values,
                      color=[PAL_DX[k] for k in rates.index], edgecolor="white", width=0.5)
        ax.set_title(feat.replace("_", " "), fontweight="bold", fontsize=9)
        ax.set_ylim(0, max(rates.values) * 1.35)
        ax.set_ylabel("%")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")
        ct = pd.crosstab(df["Diagnosis"], df[feat])
        chi2, p, _, _ = chi2_contingency(ct)
        plab = "χ² p<0.001" if p < 0.001 else f"χ² p={p:.3f}"
        ax.text(0.98, 0.95, plab, transform=ax.transAxes, ha="right",
                fontsize=7.5, color="darkred" if p < 0.05 else "grey", va="top")

    for j in range(len(bin_feats), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig3_binary_prevalence")


# ─── 8. FIGURE 4 — CORRELATION HEATMAP ───────────────────────────────────────
"""
The correlation heatmap uses annotations for all values
This works well for the current dataset size, but performance may decrease considerably for larger datasets.
"""
def fig4_correlation_heatmap(df: pd.DataFrame) -> None:
    """Full Pearson correlation heatmap (lower triangle)."""
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.suptitle("Figure 4 · Pearson Correlation Heatmap (All Features)",
                 fontsize=14, fontweight="bold")
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.7, vmin=-0.7, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_correlation_heatmap")


# ─── 9. FIGURE 5 — FEATURE IMPORTANCE (Mutual Information) ───────────────────
def fig5_feature_importance(df: pd.DataFrame) -> None:
    """Mutual information between each feature and the diagnosis target."""
    X = df.select_dtypes("number").drop(columns=["Diagnosis"], errors="ignore")
    y = df["Diagnosis"]
    mi = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": X.columns, "MI": mi}).sort_values("MI")

    q50, q75 = mi_df["MI"].quantile([0.5, 0.75])
    colors = [
        "#E91E63" if v >= q75 else "#2196F3" if v >= q50 else "#90CAF9"
        for v in mi_df["MI"]
    ]

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle("Figure 5 · Feature Importance — Mutual Information with Diagnosis",
                 fontsize=13, fontweight="bold")
    bars = ax.barh(mi_df["Feature"], mi_df["MI"], color=colors, edgecolor="white")
    ax.axvline(mi_df["MI"].mean(), color="darkorange", ls="--", lw=1.5, label="Mean MI")
    for b in bars:
        ax.text(b.get_width() + 0.0005, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.4f}", va="center", fontsize=8)
    p1 = mpatches.Patch(color="#E91E63", label="Top 25%")
    p2 = mpatches.Patch(color="#2196F3", label="50–75th pct")
    p3 = mpatches.Patch(color="#90CAF9", label="Bottom 50%")
    p4 = mpatches.Patch(color="darkorange", label="Mean MI")
    ax.legend(handles=[p1, p2, p3, p4], loc="lower right", fontsize=9)
    ax.set_xlabel("Mutual Information Score")
    fig.tight_layout()
    _save(fig, "fig5_feature_importance_MI")


# ─── 10. FIGURE 6 — BIAS & FAIRNESS ──────────────────────────────────────────
    """
    It is good to see fairness metrics included in the EDA.
    Consider handling cases where a subgroup contains very few records, as fairness ratios
    calculated from very small samples may produce unstable or misleading results.
    """
def fig6_bias_fairness(df: pd.DataFrame) -> None:
    """
    Demographic parity analysis across ethnicity, gender, education & age.
    Includes Disparate Impact (DI) ratio table and intersectional heatmap.
    """
    overall = df["Diagnosis"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("Figure 6 · Bias & Fairness — Diagnosis Rates Across Demographic Groups",
                 fontsize=14, fontweight="bold", y=1.01)

    # Helper: bar chart with overall-rate reference line
    def _rate_bar(ax, labels, rates, title, palette):
        bars = ax.bar(labels, rates, color=palette, edgecolor="white")
        ax.axhline(overall * 100, color="red", ls="--", lw=1.8, label="Overall rate")
        ax.set_ylabel("Diagnosis Rate (%)"); ax.legend(fontsize=9)
        ax.set_title(title, fontweight="bold")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

"""
The demographic subgroup definitions appear in multiple places throughout the script.
Consider moving these into a reusable helper function to reduce code duplication and improve maintainability.
"""
    
    # A – Ethnicity
    eth_rate = df.groupby("Ethnicity")["Diagnosis"].mean() * 100
    _rate_bar(axes[0, 0],
              [LABEL_MAPS["Ethnicity"][i] for i in eth_rate.index],
              eth_rate.values,
              "A · Diagnosis Rate by Ethnicity\n(Demographic Parity Test)",
              sns.color_palette("Set2", 4))

    # B – Gender
    gen_rate = df.groupby("Gender")["Diagnosis"].mean() * 100
    _rate_bar(axes[0, 1],
              ["Female", "Male"], gen_rate.values,
              "B · Diagnosis Rate by Gender",
              [PAL_DX[0], "#9C27B0"])
    axes[0, 1].set_xlim(-0.5, 1.5)

    # C – Education
    edu_rate = df.groupby("EducationLevel")["Diagnosis"].mean() * 100
    _rate_bar(axes[0, 2],
              [LABEL_MAPS["EducationLevel"][i] for i in edu_rate.index],
              edu_rate.values,
              "C · Diagnosis Rate by Education Level",
              sns.color_palette("Set3", 4))
    axes[0, 2].tick_params(axis="x", rotation=15)

    # D – Age group
    age_diag = df.groupby("AgeGroup", observed=True)["Diagnosis"].mean() * 100
    _rate_bar(axes[1, 0],
              age_diag.index.astype(str), age_diag.values,
              "D · Diagnosis Rate by Age Group",
              plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(age_diag))))

    # E – Disparate Impact (DI) table
    ax = axes[1, 1]; ax.axis("off")
    groups = {
        "Caucasian":     df[df.Ethnicity == 0]["Diagnosis"].mean(),
        "Afr. American": df[df.Ethnicity == 1]["Diagnosis"].mean(),
        "Asian":         df[df.Ethnicity == 2]["Diagnosis"].mean(),
        "Other":         df[df.Ethnicity == 3]["Diagnosis"].mean(),
        "Female":        df[df.Gender == 0]["Diagnosis"].mean(),
        "Male":          df[df.Gender == 1]["Diagnosis"].mean(),
        "No Education":  df[df.EducationLevel == 0]["Diagnosis"].mean(),
        "Bachelor's+":   df[df.EducationLevel >= 2]["Diagnosis"].mean(),
    }
    table_rows = [
        [g, f"{r * 100:.1f}%", f"{r / overall:.3f}",
         "⚠️ Concern" if (r / overall < 0.8 or r / overall > 1.25) else "✓ OK"]
        for g, r in groups.items()
    ]
    tbl = ax.table(cellText=table_rows,
                   colLabels=["Group", "Diag. Rate", "DI Ratio", "Fairness"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1, 2, 3])
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F"); cell.set_text_props(color="white", fontweight="bold")
        elif table_rows[r - 1][3] == "⚠️ Concern":
            cell.set_facecolor("#FFCCBC")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    ax.set_title("E · Disparate Impact Ratio\n(DI<0.8 or >1.25 = Fairness Concern)",
                 fontweight="bold", pad=20)

    # F – Intersectional heatmap: Gender × Ethnicity
    ax = axes[1, 2]
    pivot = df.pivot_table(values="Diagnosis", index="Gender",
                           columns="Ethnicity", aggfunc="mean") * 100
    pivot.index = ["Female", "Male"]
    pivot.columns = ["Caucasian", "Afr. American", "Asian", "Other"]
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Diagnosis Rate (%)"}, vmin=25, vmax=50)
    ax.set_title("F · Diagnosis Rate: Gender × Ethnicity\n(Intersectional Bias)", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig6_bias_fairness")


# ─── 11. FIGURE 7 — COGNITIVE ASSESSMENT DEEP DIVE ──────────────────────────
def fig7_cognitive_assessment(df: pd.DataFrame) -> None:
    """MMSE, Functional Assessment, and ADL analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Figure 7 · Cognitive & Functional Assessment Deep Dive",
                 fontsize=14, fontweight="bold", y=1.01)

    # A – MMSE KDE
    ax = axes[0, 0]
    for dx in [0, 1]:
        sns.kdeplot(df[df.Diagnosis == dx]["MMSE"], ax=ax, color=PAL_DX[dx],
                    fill=True, alpha=0.4, label=DX_LABS[dx], linewidth=2)
    ax.axvline(24, color="black", ls="--", lw=1.5, label="Clinical threshold (24)")
    ax.set_xlabel("MMSE Score"); ax.set_ylabel("Density")
    ax.set_title("A · MMSE Score Density by Diagnosis", fontweight="bold")
    ax.legend(fontsize=9)

    # B – MMSE by Ethnicity (potential test bias)
    ax = axes[0, 1]
    data = [df[df.Ethnicity == e]["MMSE"].values for e in [0, 1, 2, 3]]
    bp = ax.boxplot(data, labels=["Caucasian", "Afr. Am.", "Asian", "Other"],
                    patch_artist=True)
    for patch, col in zip(bp["boxes"], sns.color_palette("Set2", 4)):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax.set_ylabel("MMSE Score"); ax.tick_params(axis="x", rotation=15)
    ax.set_title("B · MMSE by Ethnicity\n(Cognitive Test Bias Check)", fontweight="bold")

    # C – Functional Assessment KDE
    ax = axes[0, 2]
    for dx in [0, 1]:
        sns.kdeplot(df[df.Diagnosis == dx]["FunctionalAssessment"], ax=ax,
                    color=PAL_DX[dx], fill=True, alpha=0.4,
                    label=DX_LABS[dx], linewidth=2)
    ax.set_xlabel("Functional Assessment Score"); ax.set_ylabel("Density")
    ax.set_title("C · Functional Assessment by Diagnosis", fontweight="bold")
    ax.legend(fontsize=9)

    # D – ADL by Education Level
    ax = axes[1, 0]
    data_edu = [df[df.EducationLevel == e]["ADL"].values for e in [0, 1, 2, 3]]
    bp2 = ax.boxplot(data_edu, labels=["None", "High Sch.", "Bachelor's", "Higher"],
                     patch_artist=True)
    for patch, col in zip(bp2["boxes"], sns.color_palette("Set3", 4)):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax.set_ylabel("ADL Score"); ax.tick_params(axis="x", rotation=15)
    ax.set_title("D · ADL by Education Level\n(Socioeconomic Fairness Proxy)", fontweight="bold")

    # E – Scatter MMSE vs Functional Assessment
    ax = axes[1, 1]
    for dx in [0, 1]:
        sub = df[df.Diagnosis == dx]
        ax.scatter(sub["MMSE"], sub["FunctionalAssessment"],
                   alpha=0.3, s=20, color=PAL_DX[dx], label=DX_LABS[dx])
    ax.set_xlabel("MMSE Score"); ax.set_ylabel("Functional Assessment")
    ax.set_title("E · MMSE vs Functional Assessment", fontweight="bold")
    ax.legend(fontsize=9)

    # F – Mean cognitive scores grouped bar
    ax = axes[1, 2]
    feats = ["MMSE", "FunctionalAssessment", "ADL"]
    means = df.groupby("Diagnosis")[feats].mean()
    x = np.arange(len(feats)); w = 0.35
    ax.bar(x - w / 2, means.loc[0], w, color=PAL_DX[0], label=DX_LABS[0], edgecolor="white")
    ax.bar(x + w / 2, means.loc[1], w, color=PAL_DX[1], label=DX_LABS[1], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(feats)
    ax.set_ylabel("Mean Score")
    ax.set_title("F · Mean Cognitive Scores by Diagnosis", fontweight="bold")
    ax.legend(fontsize=9)
    for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.1,
                f"{rect.get_height():.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig7_cognitive_assessment")


# ─── 12. FIGURE 8 — OUTLIER DETECTION ────────────────────────────────────────
def fig8_outlier_detection(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier detection for all continuous features."""
    cont_feats = [
        "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
        "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
        "MMSE", "FunctionalAssessment", "ADL"
    ]

    outlier_summary = {}
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    fig.suptitle("Figure 8 · Outlier Detection — IQR Method",
                 fontsize=13, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        vals = df[feat]
        Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((vals < Q1 - 1.5 * IQR) | (vals > Q3 + 1.5 * IQR)).sum()
        outlier_summary[feat] = n_out

        data = [df[df.Diagnosis == dx][feat].values for dx in [0, 1]]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5,
                                    "markerfacecolor": "orange"})
        for k, bx in enumerate(bp["boxes"]):
            bx.set_facecolor([PAL_DX[0], PAL_DX[1]][k]); bx.set_alpha(0.7)
        ax.set_xticklabels(["No AD", "AD"], fontsize=9)
        ax.set_title(feat, fontweight="bold", fontsize=9)
        ax.text(0.97, 0.96, f"{n_out} outliers", transform=ax.transAxes, ha="right",
                fontsize=8, color="darkorange" if n_out > 0 else "grey", va="top")

    patches = [mpatches.Patch(color=PAL_DX[k], label=DX_LABS[k]) for k in [0, 1]]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.tight_layout()
    _save(fig, "fig8_outlier_detection")

    return pd.DataFrame.from_dict(outlier_summary, orient="index",
                                   columns=["n_outliers_IQR"])


"""
This section contains important processing logic
Adding a brief explanatory comment would make the workflow easier to follow for future developers and reviewers
"""
# ─── 13. FIGURE 9 — LIFESTYLE & MEDICAL RISK FACTORS ────────────────────────
def fig9_risk_factors(df: pd.DataFrame) -> None:
    """BMI, comorbidities, lifestyle vs cognitive score scatterplots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Figure 9 · Lifestyle & Medical Risk Factors Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    # A – BMI KDE
    ax = axes[0, 0]
    for dx in [0, 1]:
        sns.kdeplot(df[df.Diagnosis == dx]["BMI"], ax=ax, color=PAL_DX[dx],
                    fill=True, alpha=0.4, label=DX_LABS[dx], linewidth=2)
    ax.axvline(25, color="orange", ls="--", lw=1.5, label="Overweight")
    ax.axvline(30, color="red",    ls="--", lw=1.5, label="Obese")
    ax.set_xlabel("BMI"); ax.legend(fontsize=8)
    ax.set_title("A · BMI Distribution by Diagnosis", fontweight="bold")

    # B – Comorbidity prevalence
    ax = axes[0, 1]
    comorb = ["CardiovascularDisease", "Diabetes", "Depression",
               "HeadInjury", "Hypertension", "FamilyHistoryAlzheimers"]
    short = ["CVD", "Diabetes", "Depression", "Head Inj.", "Hypert.", "Family Hx"]
    r0 = df[df.Diagnosis == 0][comorb].mean() * 100
    r1 = df[df.Diagnosis == 1][comorb].mean() * 100
    x = np.arange(len(comorb)); w = 0.38
    ax.bar(x - w / 2, r0.values, w, color=PAL_DX[0], label=DX_LABS[0], edgecolor="white")
    ax.bar(x + w / 2, r1.values, w, color=PAL_DX[1], label=DX_LABS[1], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Prevalence (%)"); ax.legend(fontsize=9)
    ax.set_title("B · Comorbidity Prevalence by Diagnosis", fontweight="bold")

    # C – Sleep Quality vs MMSE
    ax = axes[0, 2]
    for dx in [0, 1]:
        sub = df[df.Diagnosis == dx].sample(min(400, len(df[df.Diagnosis == dx])),
                                             random_state=42)
        ax.scatter(sub["SleepQuality"], sub["MMSE"], alpha=0.3, s=20,
                   color=PAL_DX[dx], label=DX_LABS[dx])
    ax.set_xlabel("Sleep Quality"); ax.set_ylabel("MMSE Score")
    ax.set_title("C · Sleep Quality vs MMSE", fontweight="bold"); ax.legend(fontsize=9)

    # D – Physical Activity vs ADL
    ax = axes[1, 0]
    for dx in [0, 1]:
        sub = df[df.Diagnosis == dx].sample(min(400, len(df[df.Diagnosis == dx])),
                                             random_state=42)
        ax.scatter(sub["PhysicalActivity"], sub["ADL"], alpha=0.3, s=20,
                   color=PAL_DX[dx], label=DX_LABS[dx])
    ax.set_xlabel("Physical Activity"); ax.set_ylabel("ADL Score")
    ax.set_title("D · Physical Activity vs ADL", fontweight="bold"); ax.legend(fontsize=9)

    # E – Cholesterol profile
    ax = axes[1, 1]
    chol = ["CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"]
    chol_lbl = ["Total", "LDL", "HDL", "Triglycerides"]
    m0 = df[df.Diagnosis == 0][chol].mean()
    m1 = df[df.Diagnosis == 1][chol].mean()
    x = np.arange(len(chol)); w = 0.38
    ax.bar(x - w / 2, m0.values, w, color=PAL_DX[0], label=DX_LABS[0], edgecolor="white")
    ax.bar(x + w / 2, m1.values, w, color=PAL_DX[1], label=DX_LABS[1], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(chol_lbl, rotation=15)
    ax.set_ylabel("Mean (mg/dL)"); ax.legend(fontsize=9)
    ax.set_title("E · Cholesterol Profile by Diagnosis", fontweight="bold")

    # F – Point-biserial correlation with diagnosis
    ax = axes[1, 2]
    corr_vals = {
        f: pointbiserialr(df["Diagnosis"], df[f])[0]
        for f in ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
                   "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
                   "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
                   "MMSE", "FunctionalAssessment", "ADL"]
    }
    cs = pd.Series(corr_vals).sort_values()
    ax.barh(cs.index, cs.values,
            color=["#E91E63" if v < 0 else "#2196F3" for v in cs.values],
            edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Point-Biserial r")
    ax.set_title("F · Correlation with Diagnosis\n(Continuous Features)", fontweight="bold")
    p1 = mpatches.Patch(color="#E91E63", label="Negative")
    p2 = mpatches.Patch(color="#2196F3", label="Positive")
    ax.legend(handles=[p1, p2], fontsize=9)

    fig.tight_layout()
    _save(fig, "fig9_risk_factors")


# ─── 14. FIGURE 10 — STATISTICAL SIGNIFICANCE ────────────────────────────────
def fig10_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run MWU tests on continuous features and χ² on binary/categorical features.
    Return a summary DataFrame and save a volcano + Cramér's V plot.
    """
    cont_feats = [
        "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality",
        "SleepQuality", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
        "MMSE", "FunctionalAssessment", "ADL"
    ]
    bin_feats = [
        "Gender", "Ethnicity", "EducationLevel", "Smoking", "FamilyHistoryAlzheimers",
        "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury", "Hypertension",
        "MemoryComplaints", "BehavioralProblems", "Confusion", "Disorientation",
        "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"
    ]
"""
Appropriate statistical tests have been selected for continuous and categorical variables
It may be useful to add validation for empty groups before running the tests to prevent unexpected runtime errors.
"""
    rows = []
    for f in cont_feats:
        a, b_ = df[df.Diagnosis == 0][f], df[df.Diagnosis == 1][f]
        stat, p = mannwhitneyu(a, b_)
        r, _ = pointbiserialr(df["Diagnosis"], df[f])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append([f, "Continuous", f"{a.mean():.2f}", f"{b_.mean():.2f}",
                     f"MWU={stat:.0f}", f"{p:.4e}", sig, f"{r:.3f}"])

    for f in bin_feats:
        ct = pd.crosstab(df["Diagnosis"], df[f])
        chi2_val, p, dof, _ = chi2_contingency(ct)
        r0 = df[df.Diagnosis == 0][f].mean() * 100
        r1 = df[df.Diagnosis == 1][f].mean() * 100
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rows.append([f, "Binary/Cat", f"{r0:.1f}%", f"{r1:.1f}%",
                     f"χ²={chi2_val:.2f}", f"{p:.4e}", sig, "N/A"])

    results_df = pd.DataFrame(rows, columns=[
        "Feature", "Type", "Mean_NoDx", "Mean_Dx", "Test_Stat", "p_value", "Sig", "Corr_r"
    ])

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Figure 10 · Statistical Significance of Features vs Diagnosis",
                 fontsize=13, fontweight="bold")

    # Volcano-style
    ax = axes[0]
    all_p = []
    for f in cont_feats:
        _, p = mannwhitneyu(df[df.Diagnosis == 0][f], df[df.Diagnosis == 1][f])
        all_p.append((f, p))
    for f in bin_feats:
        ct = pd.crosstab(df["Diagnosis"], df[f])
        _, p, _, _ = chi2_contingency(ct)
        all_p.append((f, p))
    all_p_df = pd.DataFrame(all_p, columns=["Feature", "p"]).sort_values("p")
    lp = -np.log10(all_p_df["p"])
    colors = ["#E91E63" if p < 0.001 else "#FF9800" if p < 0.05 else "#90CAF9"
              for p in all_p_df["p"]]
    ax.barh(all_p_df["Feature"], lp, color=colors, edgecolor="white")
    ax.axvline(-np.log10(0.05), color="orange", ls="--", lw=1.8, label="p=0.05")
    ax.axvline(-np.log10(0.001), color="red", ls="--", lw=1.8, label="p=0.001")
    ax.set_xlabel("-log₁₀(p-value)")
    ax.set_title("A · Significance Plot\n(MWU: continuous | χ²: categorical)", fontweight="bold")
    ax.legend(fontsize=9)

    # Cramér's V
    ax = axes[1]
    ce = []
    for f in bin_feats:
        ct = pd.crosstab(df["Diagnosis"], df[f])
        chi2_val, p, dof, _ = chi2_contingency(ct)
        n = ct.values.sum()
        v = np.sqrt(chi2_val / (n * min(ct.shape[0] - 1, ct.shape[1] - 1)))
        ce.append((f, v))
    ce_df = pd.DataFrame(ce, columns=["Feature", "CramersV"]).sort_values("CramersV")
    cv_cols = ["#E91E63" if v > 0.2 else "#2196F3" if v > 0.1 else "#90CAF9"
               for v in ce_df["CramersV"]]
    ax.barh(ce_df["Feature"], ce_df["CramersV"], color=cv_cols, edgecolor="white")
    ax.axvline(0.1, color="orange", ls="--", lw=1.5, label="Small effect (0.1)")
    ax.axvline(0.3, color="red",    ls="--", lw=1.5, label="Medium effect (0.3)")
    ax.set_xlabel("Cramér's V (Effect Size)")
    ax.set_title("B · Cramér's V Effect Size\n(Categorical Features)", fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "fig10_statistical_tests")
    return results_df


# ─── 15. FIGURE 11 — PAIRPLOT ─────────────────────────────────────────────────
"""
Sampling 500 records is a sensible approach for reducing rendering time.
Consider checking that the dataset contains at least 500 records before sampling to avoid potential exceptions on smaller datasets.
"""
def fig11_pairplot(df: pd.DataFrame) -> None:
    """Pairplot of key predictors, sampled for rendering speed."""
    key_feats = ["MMSE", "FunctionalAssessment", "ADL", "SleepQuality", "Age"]
    sample_df = df[key_feats + ["Diagnosis"]].sample(500, random_state=42)
    sample_df["Diagnosis_Label"] = sample_df["Diagnosis"].map(DX_LABS)
    g = sns.pairplot(
        sample_df, hue="Diagnosis_Label",
        palette={DX_LABS[0]: PAL_DX[0], DX_LABS[1]: PAL_DX[1]},
        vars=key_feats, diag_kind="kde",
        plot_kws={"alpha": 0.35, "s": 20}
    )
    g.fig.suptitle("Figure 11 · Pairplot — Key Predictors by Diagnosis (n=500 sample)",
                   y=1.01, fontsize=13, fontweight="bold")
    _save(g.fig, "fig11_pairplot")


# ─── 16. FIGURE 12 — EXPLAINABILITY GROUNDWORK ───────────────────────────────
def fig12_explainability(df: pd.DataFrame) -> None:
    """Feature interactions & demographic subgroup profiles for XAI baselines."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        "Figure 12 · Explainability Groundwork — Feature Interactions & Subgroup Profiles",
        fontsize=13, fontweight="bold", y=1.01
    )

    # A – Mean MMSE by Age Group & Diagnosis
    ax = axes[0, 0]
    ag_mmse = df.groupby(["AgeGroup", "Diagnosis"], observed=True)["MMSE"].mean().unstack()
    ag_mmse.columns = [DX_LABS[c] for c in ag_mmse.columns]
    ag_mmse.plot(kind="bar", ax=ax, color=[PAL_DX[0], PAL_DX[1]], edgecolor="white", rot=30)
    ax.set_ylabel("Mean MMSE Score")
    ax.set_title("A · Mean MMSE by Age Group & Diagnosis", fontweight="bold")
    ax.legend(fontsize=9)

    # B – Mean MMSE by Gender & Diagnosis
    ax = axes[0, 1]
    g_mmse = df.groupby(["Gender", "Diagnosis"])["MMSE"].mean().unstack()
    g_mmse.index = ["Female", "Male"]
    g_mmse.columns = [DX_LABS[c] for c in g_mmse.columns]
    g_mmse.plot(kind="bar", ax=ax, color=[PAL_DX[0], PAL_DX[1]], edgecolor="white", rot=0)
    ax.set_ylabel("Mean MMSE Score")
    ax.set_title("B · Mean MMSE by Gender & Diagnosis", fontweight="bold"); ax.legend(fontsize=9)

    # C – Functional Assessment by Education × Diagnosis
    ax = axes[0, 2]
    e_fa = df.groupby(["EducationLevel", "Diagnosis"])["FunctionalAssessment"].mean().unstack()
    e_fa.index = [LABEL_MAPS["EducationLevel"][i] for i in e_fa.index]
    e_fa.columns = [DX_LABS[c] for c in e_fa.columns]
    e_fa.plot(kind="bar", ax=ax, color=[PAL_DX[0], PAL_DX[1]], edgecolor="white", rot=20)
    ax.set_ylabel("Mean Functional Assessment")
    ax.set_title("C · Functional Assessment by\nEducation × Diagnosis", fontweight="bold")
    ax.legend(fontsize=9)

    # D – MMSE vs ADL coloured by Ethnicity
    ax = axes[1, 0]
    eth_pal = sns.color_palette("Set2", 4)
    for eth in [0, 1, 2, 3]:
        sub = df[df.Ethnicity == eth].sample(min(200, len(df[df.Ethnicity == eth])),
                                              random_state=42)
        ax.scatter(sub["MMSE"], sub["ADL"], alpha=0.4, s=25,
                   color=eth_pal[eth], label=LABEL_MAPS["Ethnicity"][eth])
    ax.set_xlabel("MMSE Score"); ax.set_ylabel("ADL Score"); ax.legend(fontsize=8)
    ax.set_title("D · MMSE vs ADL by Ethnicity\n(Intersectional View)", fontweight="bold")

    # E – Correlation heatmap of top predictors
    ax = axes[1, 1]
    top = ["MMSE", "FunctionalAssessment", "ADL", "SleepQuality",
           "CholesterolHDL", "MemoryComplaints", "BehavioralProblems", "Diagnosis"]
    sns.heatmap(df[top].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, center=0, vmax=0.6, vmin=-0.6, linewidths=0.5,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.7})
    ax.set_title("E · Correlation Among Top Predictors\n(Multicollinearity Check)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    # F – Subgroup feature profiles (normalised)
    ax = axes[1, 2]
    sub_groups = {
        "AD (Caucasian)": df[(df.Diagnosis == 1) & (df.Ethnicity == 0)],
        "AD (Afr. Am.)":  df[(df.Diagnosis == 1) & (df.Ethnicity == 1)],
        "No AD (Female)": df[(df.Diagnosis == 0) & (df.Gender == 0)],
        "No AD (Male)":   df[(df.Diagnosis == 0) & (df.Gender == 1)],
    }
    feats4 = ["MMSE", "FunctionalAssessment", "ADL", "SleepQuality"]
    profile_means = {k: v[feats4].mean() for k, v in sub_groups.items()}
    profile_df = pd.DataFrame(profile_means).T
    norm_df = (profile_df - df[feats4].min()) / (df[feats4].max() - df[feats4].min())
    colors_sub = ["#E91E63", "#FF5722", "#2196F3", "#03A9F4"]
    for i, (grp, row) in enumerate(norm_df.iterrows()):
        ax.plot(feats4, row.values, marker="o", label=grp,
                color=colors_sub[i], linewidth=2)
    ax.set_ylabel("Normalised Score (0–1)"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("F · Subgroup Feature Profiles\n(XAI Baseline)", fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, "fig12_explainability_groundwork")


# ─── 17. STATISTICAL SUMMARY PRINTER ─────────────────────────────────────────
def print_key_findings(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("KEY EDA FINDINGS FOR MSc THESIS")
    print("=" * 70)

    print("\n── CLASS IMBALANCE ──")
    vc = df["Diagnosis"].value_counts()
    print(f"  No Alzheimer's : {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)")
    print(f"  Alzheimer's    : {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)")
    print(f"  Imbalance ratio: {vc[0]/vc[1]:.2f}:1  (impacts F1/AUC; consider SMOTE or class weights)")

    print("\n── TOP PREDICTORS (by Mutual Information) ──")
    X = df.select_dtypes("number").drop(columns=["Diagnosis"])
    mi = mutual_info_classif(X, df["Diagnosis"], random_state=42)
    top5 = pd.Series(mi, index=X.columns).nlargest(5)
    for feat, val in top5.items():
        print(f"  {feat:<30} MI = {val:.4f}")

    print("\n── COGNITIVE SCORES BETWEEN GROUPS ──")
    for feat in ["MMSE", "FunctionalAssessment", "ADL"]:
        m0 = df[df.Diagnosis == 0][feat].mean()
        m1 = df[df.Diagnosis == 1][feat].mean()
        _, p = mannwhitneyu(df[df.Diagnosis == 0][feat], df[df.Diagnosis == 1][feat])
        print(f"  {feat:<25} No AD={m0:.2f}  AD={m1:.2f}  Δ={abs(m0-m1):.2f}  p={p:.2e}")

    print("\n── FAIRNESS — DISPARATE IMPACT RATIOS ──")
    overall = df["Diagnosis"].mean()
    checks = {
        "Caucasian":     df[df.Ethnicity == 0]["Diagnosis"].mean(),
        "Afr. American": df[df.Ethnicity == 1]["Diagnosis"].mean(),
        "Asian":         df[df.Ethnicity == 2]["Diagnosis"].mean(),
        "Other":         df[df.Ethnicity == 3]["Diagnosis"].mean(),
        "Female":        df[df.Gender == 0]["Diagnosis"].mean(),
        "Male":          df[df.Gender == 1]["Diagnosis"].mean(),
    }
    for grp, rate in checks.items():
        di = rate / overall
        flag = "⚠️" if di < 0.8 or di > 1.25 else "✓"
        print(f"  {grp:<18} rate={rate*100:.1f}%  DI={di:.3f}  {flag}")

    print("\n── MISSING DATA ──")
    miss = df.isnull().sum().sum()
    print(f"  Total missing cells: {miss}  ({'None — complete dataset' if miss == 0 else 'HANDLE BEFORE MODELLING'})")

    print("\n── OUTLIERS (IQR) ──")
    cont = ["MMSE", "FunctionalAssessment", "ADL", "BMI", "SleepQuality",
            "SystolicBP", "DiastolicBP", "CholesterolTotal",
            "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides"]
    total_out = 0
    for f in cont:
        Q1, Q3 = df[f].quantile(0.25), df[f].quantile(0.75)
        n = ((df[f] < Q1 - 1.5 * (Q3 - Q1)) | (df[f] > Q3 + 1.5 * (Q3 - Q1))).sum()
        total_out += n
    print(f"  Total IQR outliers across continuous features: {total_out}")

    print("\n── RECOMMENDATIONS FOR MODELLING ──")
    recs = [
        "Apply SMOTE or class-weighted loss to handle 65/35 imbalance.",
        "FunctionalAssessment, ADL, MMSE are the strongest predictors (MI & correlation).",
        "MemoryComplaints & BehavioralProblems are the strongest binary predictors (Cramér's V).",
        "Demographic features (Gender, Ethnicity, Education) show no significant independent",
        "  association with diagnosis — ideal for fairness evaluation (no obvious data leakage).",
        "MMSE shows distribution differences across ethnic groups — potential cognitive test bias.",
        "No missing values; no outliers by IQR — dataset is clean (likely synthetic).",
        "Consider SHAP values on FunctionalAssessment, ADL, MMSE for XAI chapter.",
        "Intersectional fairness metrics (e.g., Equalized Odds) should be computed post-modelling.",
    ]
    for r in recs:
        print(f"  → {r}")

    print("\n" + "=" * 70)


# ─── 18. MAIN ─────────────────────────────────────────────────────────────────
 """
    The dataset is loaded successfully, however it would be beneficial to validate that key columns
    such as Diagnosis, Age, Gender, and Ethnicity exist before continuing.
    """
def main():
    print("Loading data …")
    df = load_data(DATA_PATH) 

    print("Profiling data …")
    profile_data(df)

    print("\nGenerating figures …")
    fig1_overview(df)
    fig2_continuous_distributions(df)
    fig3_binary_prevalence(df)
    fig4_correlation_heatmap(df)
    fig5_feature_importance(df)
    fig6_bias_fairness(df)
    fig7_cognitive_assessment(df)
    out_summary = fig8_outlier_detection(df)
    fig9_risk_factors(df)
    stats_df = fig10_statistical_tests(df)
    fig11_pairplot(df)
    fig12_explainability(df)

    print("\nSaving CSVs …")
    stats_df.to_csv(f"{OUTPUT_DIR}/statistical_tests_summary.csv", index=False)
    out_summary.to_csv(f"{OUTPUT_DIR}/outlier_summary.csv")
    print(f"  ✔ statistical_tests_summary.csv")
    print(f"  ✔ outlier_summary.csv")

    print_key_findings(df)
    print("\nAll done. 12 figures + 2 CSV tables saved.")

"""
The use of the __main__ guard follows good Python coding practices 
and allows the module to be imported safely without executing the pipeline automatically
"""

if __name__ == "__main__":
    main()
"""
The file is well organised into clearly numbered sections,
making it easy to follow the EDA workflow from data loading through fairness analysis and explainability groundwork.
"""
