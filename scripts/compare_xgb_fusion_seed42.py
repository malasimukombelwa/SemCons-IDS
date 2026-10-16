import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NB = Path(
    "notebooks/core/"
    "SemCons_IDS_MiniLM_ModernBERT_Comparative_Final_v8.ipynb"
)

DATA = Path(
    "/project/def-missro01/malj30/datasets/CICIDS2017"
)

RUN = Path(
    "/project/def-missro01/malj30/semcons_ids/models/"
    "temporal_within_file__minilm__seed_42"
)

nb = json.loads(NB.read_text())

# Extraire les fonctions originales de nettoyage
# sans exécuter toute la cellule Colab.
src6 = "".join(nb["cells"][6]["source"])
clean_functions = src6.split("files = find_data_files")[0]

scope = {
    "pd": pd,
    "np": np,
    "re": re,
    "Path": Path,
    "List": List,
}

exec(clean_functions, scope)

cfg = SimpleNamespace(
    data_dir=str(DATA),
    min_samples_per_class=200,
)

files = scope["find_data_files"](cfg.data_dir)

assert len(files) == 8, f"8 fichiers attendus, {len(files)} trouvés"

frames = [scope["read_one_file"](p) for p in files]
df = pd.concat(frames, ignore_index=True)

label_candidates = [
    c for c in df.columns
    if c == "label" or "label" in c or "class" in c
]
assert label_candidates
label_col = label_candidates[0]

df[label_col] = df[label_col].map(scope["normalize_label"])
df = df.dropna(subset=[label_col]).copy()

dedup_cols = [
    c for c in df.columns
    if c not in {"__source_file__", "__row_in_file__"}
]
df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

features = [
    c for c in df.columns
    if c not in {label_col, "__source_file__", "__row_in_file__"}
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

counts = df[label_col].value_counts()
keep = counts[counts >= cfg.min_samples_per_class].index
df = df[df[label_col].isin(keep)].reset_index(drop=True)

print("Observations nettoyées :", len(df))
print("Caractéristiques :", len(features))

assert len(df) == 2520172, (
    f"Effectif différent de Colab : {len(df)}"
)

# Extraire uniquement la fonction originale make_splits.
src10 = "".join(nb["cells"][10]["source"])
split_code = src10.split(
    "def capped_indices_per_class"
)[0]

scope.update({
    "math": math,
    "Dict": Dict,
    "train_test_split": train_test_split,
})

exec(split_code, scope)

splits = scope["make_splits"](
    data=df,
    protocol="temporal_within_file",
    seed=42,
    label_col=label_col,
    train_fraction=0.70,
    val_fraction=0.10,
    cal_fraction=0.10,
    test_fraction=0.10,
)

test = splits["test"]

print("Observations test :", len(test))

saved = pd.read_csv(
    RUN / "test_predictions.csv",
    usecols=["true_label"],
)

expected = saved["true_label"].astype(str).to_numpy()
actual = test[label_col].astype(str).to_numpy()

assert len(actual) == len(expected), (
    f"Effectifs différents : {len(actual)} / {len(expected)}"
)

matches = actual == expected
print("Étiquettes alignées :", int(matches.sum()), "/", len(matches))
print("Taux de correspondance :", float(matches.mean()))

assert matches.all(), (
    "Les étiquettes ne sont pas alignées. "
    "Ne pas lancer la comparaison XGBoost."
)

print("===== CONTRÔLE DES ÉTIQUETTES RÉUSSI =====")

# Sauvegarder les identifiants originaux des lignes.
out = Path("reports/crisis_analysis")
out.mkdir(parents=True, exist_ok=True)

test[[
    "__source_file__", "__row_in_file__", label_col
]].to_csv(
    out / "reconstructed_temporal_minilm_seed42_test_ids.csv",
    index=False,
)

print("Identifiants reconstruits sauvegardés")

# ============================================================
# Réinférence XGBoost et comparaison avec le système final
# ============================================================

import sys
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, accuracy_score
)

sys.path.insert(0, str(Path("src").resolve()))
from semcons_corrected_pipeline_utils import (
    load_run_artifacts,
    transform_with_saved_preprocessor,
)

print("\n===== CHARGEMENT XGBOOST =====", flush=True)

art = load_run_artifacts(RUN)

# Inférence par blocs pour limiter la mémoire.
batch_size = 20000
predicted_ids = []

for start in range(0, len(test), batch_size):
    batch = test.iloc[start:start + batch_size].copy()
    X = transform_with_saved_preprocessor(batch, art)
    probabilities = art.xgb_model.predict_proba(X["raw_numeric"])
    predicted_ids.append(
        np.asarray(probabilities).argmax(axis=1)
    )
    print(
        f"XGBoost : {min(start + batch_size, len(test)):,}"
        f"/{len(test):,}",
        flush=True,
    )

xgb_ids = np.concatenate(predicted_ids)
xgb_labels = art.label_encoder.inverse_transform(xgb_ids)

saved_final = pd.read_csv(
    RUN / "test_predictions.csv",
    usecols=["true_label", "predicted_label"],
)

y_true = saved_final["true_label"].astype(str).to_numpy()
y_final = saved_final["predicted_label"].astype(str).to_numpy()
y_xgb = np.asarray(xgb_labels).astype(str)

assert np.array_equal(
    y_true, test[label_col].astype(str).to_numpy()
), "Désalignement des étiquettes"

classes = list(art.label_encoder.classes_)
present = [c for c in classes if np.any(y_true == c)]

def show_metrics(name, prediction):
    print(f"\n===== {name} =====")
    print("Accuracy :", round(accuracy_score(y_true, prediction), 6))
    print(
        "Macro-F1 classes présentes :",
        round(f1_score(
            y_true, prediction, labels=present,
            average="macro", zero_division=0
        ), 6)
    )
    print(
        "Macro-F1 10 classes :",
        round(f1_score(
            y_true, prediction, labels=classes,
            average="macro", zero_division=0
        ), 6)
    )

show_metrics("XGBOOST RECONSTRUIT", y_xgb)
show_metrics("STACKING SAUVEGARDÉ", y_final)

rows = []
for cls in classes:
    mask = y_true == cls

    p_x, r_x, f_x, _ = precision_recall_fscore_support(
        y_true, y_xgb, labels=[cls], zero_division=0
    )
    p_f, r_f, f_f, _ = precision_recall_fscore_support(
        y_true, y_final, labels=[cls], zero_division=0
    )

    corrected = mask & (y_xgb != y_true) & (y_final == y_true)
    introduced = mask & (y_xgb == y_true) & (y_final != y_true)

    rows.append({
        "class": cls,
        "support": int(mask.sum()),
        "xgb_precision": float(p_x[0]),
        "xgb_recall": float(r_x[0]),
        "xgb_f1": float(f_x[0]),
        "final_precision": float(p_f[0]),
        "final_recall": float(r_f[0]),
        "final_f1": float(f_f[0]),
        "delta_f1": float(f_f[0] - f_x[0]),
        "xgb_errors": int((mask & (y_xgb != y_true)).sum()),
        "final_errors": int((mask & (y_final != y_true)).sum()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
    })

comparison = pd.DataFrame(rows)
comparison.to_csv(
    out / "xgb_vs_fusion_temporal_minilm_seed42.csv",
    index=False
)

print("\n===== COMPARAISON PAR CLASSE =====")
print(
    comparison[
        ["class", "support", "xgb_f1", "final_f1",
         "delta_f1", "xgb_errors", "final_errors",
         "corrected", "introduced"]
    ].round(5).to_string(index=False)
)

print("\n===== BILAN GLOBAL APPARIÉ =====")
print(
    "Erreurs corrigées :",
    int(((y_xgb != y_true) & (y_final == y_true)).sum())
)
print(
    "Erreurs introduites :",
    int(((y_xgb == y_true) & (y_final != y_true)).sum())
)
print(
    "Erreurs communes :",
    int(((y_xgb != y_true) & (y_final != y_true)).sum())
)

print("\n===== RÉFÉRENCES HISTORIQUES COLAB =====")
for filename in ["branch_summary.csv", "candidate_test_comparison.csv"]:
    ref = pd.read_csv(RUN / filename)
    print(f"\n{filename}")
    print("Colonnes :", list(ref.columns))
    if len(ref) <= 20:
        print(ref.to_string(index=False))
    else:
        print(ref.head(12).to_string(index=False))

print("\nAnalyse terminée.")
