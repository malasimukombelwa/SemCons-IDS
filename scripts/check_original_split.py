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
