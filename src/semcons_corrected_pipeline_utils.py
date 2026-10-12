"""Utilities for corrected SemCons/SemFuse re-analysis and external validation.

The module is designed for Google Colab and the artifacts produced by
SemCons_IDS_MiniLM_ModernBERT_Comparative_Final_v8.

Scientific safeguards implemented here:
- source preprocessing is loaded from each saved run and never refit on test data;
- the original calibration partition is split into four disjoint parts:
  meta-train, model-selection, temperature-calibration, and conformal-calibration;
- the stacker is trained only on meta-train;
- candidate selection uses model-selection and defaults to observed-class Macro-F1;
- temperature and conformal thresholds are fitted on separate partitions;
- the final test partition remains untouched;
- external validation distinguishes common-class closed-set evaluation from
  binary attack-vs-benign evaluation on unseen attack families.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.optimize import minimize_scalar
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EPS = 1e-12
UTILS_VERSION = "2026.07.26-v6-self-contained-resumable"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_column_name(name: object) -> str:
    value = str(name).strip().lower()
    value = value.replace("/s", "_s")
    value = re.sub(r"[\s/\\-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


SOURCE_LABEL_MAP = {
    "benign": "BENIGN",
    "bot": "Bot",
    "ddos": "DDoS",
    "dos hulk": "DoS_Hulk",
    "dos goldeneye": "DoS_GoldenEye",
    "dos slowhttptest": "DoS_Slowhttptest",
    "dos slowloris": "DoS_Slowloris",
    "ftp patator": "FTP-Patator",
    "ssh patator": "SSH-Patator",
    "portscan": "PortScan",
    "port scan": "PortScan",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
}


def normalize_source_label(value: object) -> Optional[str]:
    key = str(value).strip().lower().replace("_", "-")
    key = re.sub(r"[-\s]+", " ", key).strip()
    return SOURCE_LABEL_MAP.get(key)


def normalize_target_label(value: object) -> Tuple[Optional[str], bool]:
    """Map CSE-CIC-IDS2018 labels to the CICIDS2017 source taxonomy.

    Returns (mapped_source_label, is_attack). mapped_source_label is None for
    attack families that are not represented in the source 10-class head.
    """
    raw = str(value).strip()
    key = re.sub(r"[_\-\s]+", " ", raw.lower()).strip()
    is_benign = "benign" in key
    if is_benign:
        return "BENIGN", False
    if key == "bot" or "botnet" in key:
        return "Bot", True
    if "ddos" in key or "distributed denial" in key:
        return "DDoS", True
    if "goldeneye" in key:
        return "DoS_GoldenEye", True
    if re.search(r"\bhulk\b", key):
        return "DoS_Hulk", True
    if "slowhttptest" in key or "slow http test" in key:
        return "DoS_Slowhttptest", True
    if "slowloris" in key:
        return "DoS_Slowloris", True
    if "ftp" in key and ("brute" in key or "patator" in key):
        return "FTP-Patator", True
    if "ssh" in key and ("brute" in key or "patator" in key):
        return "SSH-Patator", True
    if "portscan" in key or "port scan" in key:
        return "PortScan", True
    # Web attacks, SQL injection, infiltration, and other target-only families
    # remain unseen for the source multiclass head.
    return None, True


# Common CSE-CIC-IDS2018 / CICFlowMeter abbreviations mapped to the canonical
# names used by the saved CICIDS2017 pipeline.
FEATURE_ALIASES = {
    "dst_port": "destination_port",
    "destination_port": "destination_port",
    "flow_duration": "flow_duration",
    "tot_fwd_pkts": "total_fwd_packets",
    "total_fwd_packets": "total_fwd_packets",
    "tot_bwd_pkts": "total_backward_packets",
    "total_backward_packets": "total_backward_packets",
    "totlen_fwd_pkts": "total_length_of_fwd_packets",
    "total_length_of_fwd_packets": "total_length_of_fwd_packets",
    "totlen_bwd_pkts": "total_length_of_bwd_packets",
    "total_length_of_bwd_packets": "total_length_of_bwd_packets",
    "fwd_pkt_len_max": "fwd_packet_length_max",
    "fwd_packet_length_max": "fwd_packet_length_max",
    "fwd_pkt_len_min": "fwd_packet_length_min",
    "fwd_packet_length_min": "fwd_packet_length_min",
    "fwd_pkt_len_mean": "fwd_packet_length_mean",
    "fwd_packet_length_mean": "fwd_packet_length_mean",
    "fwd_pkt_len_std": "fwd_packet_length_std",
    "fwd_packet_length_std": "fwd_packet_length_std",
    "bwd_pkt_len_max": "bwd_packet_length_max",
    "bwd_packet_length_max": "bwd_packet_length_max",
    "bwd_pkt_len_min": "bwd_packet_length_min",
    "bwd_packet_length_min": "bwd_packet_length_min",
    "bwd_pkt_len_mean": "bwd_packet_length_mean",
    "bwd_packet_length_mean": "bwd_packet_length_mean",
    "bwd_pkt_len_std": "bwd_packet_length_std",
    "bwd_packet_length_std": "bwd_packet_length_std",
    "flow_byts_s": "flow_bytes_s",
    "flow_bytes_s": "flow_bytes_s",
    "flow_pkts_s": "flow_packets_s",
    "flow_packets_s": "flow_packets_s",
    "flow_iat_mean": "flow_iat_mean",
    "flow_iat_std": "flow_iat_std",
    "flow_iat_max": "flow_iat_max",
    "flow_iat_min": "flow_iat_min",
    "fwd_iat_tot": "fwd_iat_total",
    "fwd_iat_total": "fwd_iat_total",
    "fwd_iat_mean": "fwd_iat_mean",
    "fwd_iat_std": "fwd_iat_std",
    "fwd_iat_max": "fwd_iat_max",
    "fwd_iat_min": "fwd_iat_min",
    "bwd_iat_tot": "bwd_iat_total",
    "bwd_iat_total": "bwd_iat_total",
    "bwd_iat_mean": "bwd_iat_mean",
    "bwd_iat_std": "bwd_iat_std",
    "bwd_iat_max": "bwd_iat_max",
    "bwd_iat_min": "bwd_iat_min",
    "fwd_psh_flags": "fwd_psh_flags",
    "bwd_psh_flags": "bwd_psh_flags",
    "fwd_urg_flags": "fwd_urg_flags",
    "bwd_urg_flags": "bwd_urg_flags",
    "fwd_header_len": "fwd_header_length",
    "fwd_header_length": "fwd_header_length",
    "fwd_header_len_1": "fwd_header_length1",
    "fwd_header_length1": "fwd_header_length1",
    "bwd_header_len": "bwd_header_length",
    "bwd_header_length": "bwd_header_length",
    "fwd_pkts_s": "fwd_packets_s",
    "fwd_packets_s": "fwd_packets_s",
    "bwd_pkts_s": "bwd_packets_s",
    "bwd_packets_s": "bwd_packets_s",
    "pkt_len_min": "min_packet_length",
    "min_packet_length": "min_packet_length",
    "pkt_len_max": "max_packet_length",
    "max_packet_length": "max_packet_length",
    "pkt_len_mean": "packet_length_mean",
    "packet_length_mean": "packet_length_mean",
    "pkt_len_std": "packet_length_std",
    "packet_length_std": "packet_length_std",
    "pkt_len_var": "packet_length_variance",
    "packet_length_variance": "packet_length_variance",
    "fin_flag_cnt": "fin_flag_count",
    "fin_flag_count": "fin_flag_count",
    "syn_flag_cnt": "syn_flag_count",
    "syn_flag_count": "syn_flag_count",
    "rst_flag_cnt": "rst_flag_count",
    "rst_flag_count": "rst_flag_count",
    "psh_flag_cnt": "psh_flag_count",
    "psh_flag_count": "psh_flag_count",
    "ack_flag_cnt": "ack_flag_count",
    "ack_flag_count": "ack_flag_count",
    "urg_flag_cnt": "urg_flag_count",
    "urg_flag_count": "urg_flag_count",
    "cwe_flag_count": "cwe_flag_count",
    "ece_flag_cnt": "ece_flag_count",
    "ece_flag_count": "ece_flag_count",
    "down_up_ratio": "down_up_ratio",
    "pkt_size_avg": "average_packet_size",
    "average_packet_size": "average_packet_size",
    "fwd_seg_size_avg": "avg_fwd_segment_size",
    "avg_fwd_segment_size": "avg_fwd_segment_size",
    "bwd_seg_size_avg": "avg_bwd_segment_size",
    "avg_bwd_segment_size": "avg_bwd_segment_size",
    "fwd_byts_b_avg": "fwd_avg_bytes_bulk",
    "fwd_pkts_b_avg": "fwd_avg_packets_bulk",
    "fwd_blk_rate_avg": "fwd_avg_bulk_rate",
    "bwd_byts_b_avg": "bwd_avg_bytes_bulk",
    "bwd_pkts_b_avg": "bwd_avg_packets_bulk",
    "bwd_blk_rate_avg": "bwd_avg_bulk_rate",
    "subflow_fwd_pkts": "subflow_fwd_packets",
    "subflow_fwd_packets": "subflow_fwd_packets",
    "subflow_fwd_byts": "subflow_fwd_bytes",
    "subflow_fwd_bytes": "subflow_fwd_bytes",
    "subflow_bwd_pkts": "subflow_bwd_packets",
    "subflow_bwd_packets": "subflow_bwd_packets",
    "subflow_bwd_byts": "subflow_bwd_bytes",
    "subflow_bwd_bytes": "subflow_bwd_bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_win_bytes_forward": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "init_win_bytes_backward": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "min_seg_size_forward": "min_seg_size_forward",
    "active_mean": "active_mean",
    "active_std": "active_std",
    "active_max": "active_max",
    "active_min": "active_min",
    "idle_mean": "idle_mean",
    "idle_std": "idle_std",
    "idle_max": "idle_max",
    "idle_min": "idle_min",
}


def canonicalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    used = set()
    for original in frame.columns:
        normalized = normalize_column_name(original)
        canonical = FEATURE_ALIASES.get(normalized, normalized)
        # Avoid silently overwriting two distinct source columns.
        if canonical in used:
            suffix = 1
            candidate = f"{canonical}{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{canonical}{suffix}"
            canonical = candidate
        rename[original] = canonical
        used.add(canonical)
    return frame.rename(columns=rename)


def find_data_files(data_dir: str | Path) -> List[Path]:
    root = Path(data_dir)
    files: List[Path] = []
    for pattern in ("**/*.parquet", "**/*.csv", "**/*.feather"):
        files.extend(root.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def read_frame(path: Path, use_feature_aliases: bool = True) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".feather":
        frame = pd.read_feather(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file: {path}")
    if use_feature_aliases:
        frame = canonicalize_feature_columns(frame)
    else:
        frame.columns = [normalize_column_name(c) for c in frame.columns]
    frame["__source_file__"] = path.name
    frame["__row_in_file__"] = np.arange(len(frame), dtype=np.int64)
    return frame


def identify_label_column(frame: pd.DataFrame) -> str:
    candidates = [c for c in frame.columns if c == "label" or "label" in c or c in {"class", "attack_cat"}]
    if not candidates:
        raise ValueError("No label column found.")
    return candidates[0]


def load_cicids2017(
    data_dir: str | Path,
    min_samples_per_class: int = 200,
    dedup_mode: str = "original",
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    files = find_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet/Feather found in {data_dir}")
    frames = [read_frame(path, use_feature_aliases=False) for path in files]
    data = pd.concat(frames, ignore_index=True)
    label_col = identify_label_column(data)
    data[label_col] = data[label_col].map(normalize_source_label)
    unmapped = int(data[label_col].isna().sum())
    data = data.dropna(subset=[label_col]).copy()

    metadata_cols = {label_col, "__source_file__", "__row_in_file__"}
    feature_cols = [c for c in data.columns if c not in metadata_cols]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Duplicate audit on feature vectors, independent of labels.
    feature_hash = pd.util.hash_pandas_object(data[feature_cols], index=False)
    audit = data[[label_col, "__source_file__", "__row_in_file__"]].copy()
    audit["__feature_hash__"] = feature_hash.values
    conflict_counts = audit.groupby("__feature_hash__")[label_col].nunique()
    conflict_hashes = set(conflict_counts[conflict_counts > 1].index)
    audit["label_conflict"] = audit["__feature_hash__"].isin(conflict_hashes)
    conflicts = audit[audit["label_conflict"]].copy()

    # Saved-model re-analysis must reproduce the exact cleaning used during
    # training. The original pipeline deduplicated all feature columns plus the
    # label, while excluding only source-file metadata. A stricter feature-only
    # mode is available for future retraining, but it must not be mixed with the
    # already saved models.
    before = len(data)
    if dedup_mode == "original":
        dedup_columns = [c for c in data.columns if c not in {"__source_file__", "__row_in_file__"}]
        data = data.drop_duplicates(subset=dedup_columns, keep="first").copy()
    elif dedup_mode == "feature_only_strict":
        if conflict_hashes:
            data = data.loc[~feature_hash.isin(conflict_hashes)].copy()
            feature_hash = feature_hash.loc[data.index]
        data["__feature_hash__"] = feature_hash.values
        data = data.drop_duplicates(subset=["__feature_hash__"], keep="first").copy()
        data = data.drop(columns=["__feature_hash__"])
    else:
        raise ValueError("dedup_mode must be 'original' or 'feature_only_strict'.")
    dedup_removed = before - len(data)
    data = data.reset_index(drop=True)

    counts = data[label_col].value_counts()
    keep = counts[counts >= min_samples_per_class].index
    data = data[data[label_col].isin(keep)].reset_index(drop=True)
    report = pd.DataFrame({
        "item": ["files", "unmapped_labels", "label_conflict_rows", "deduplicated_rows", "retained_rows", "retained_classes"],
        "value": [len(files), unmapped, len(conflicts), dedup_removed, len(data), data[label_col].nunique()],
    })
    return data, label_col, report


def load_cse_cic_ids2018(data_dir: str | Path) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    files = find_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No target dataset files found in {data_dir}")
    frames = []
    for path in files:
        frame = read_frame(path, use_feature_aliases=True)
        label_col = identify_label_column(frame)
        if label_col != "label":
            frame = frame.rename(columns={label_col: "label"})
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    mapped = data["label"].map(normalize_target_label)
    data["__mapped_label__"] = [x[0] for x in mapped]
    data["__is_attack__"] = [x[1] for x in mapped]
    data["__raw_label__"] = data["label"].astype(str)

    metadata = {"label", "__mapped_label__", "__is_attack__", "__raw_label__", "__source_file__", "__row_in_file__"}
    for col in [c for c in data.columns if c not in metadata]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    feature_cols = [c for c in data.columns if c not in metadata]
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    audit = (
        data.groupby(["__raw_label__", "__mapped_label__", "__is_attack__"], dropna=False)
        .size().rename("count").reset_index().sort_values("count", ascending=False)
    )
    return data, "label", audit


def make_original_source_splits(
    data: pd.DataFrame,
    protocol: str,
    seed: int,
    label_col: str,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    cal_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> Dict[str, pd.DataFrame]:
    total = train_fraction + val_fraction + cal_fraction + test_fraction
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.")
    if protocol == "iid_stratified":
        train, temp = train_test_split(
            data,
            test_size=1.0 - train_fraction,
            stratify=data[label_col],
            random_state=seed,
        )
        rel_val = val_fraction / (val_fraction + cal_fraction + test_fraction)
        val, remain = train_test_split(
            temp,
            test_size=1.0 - rel_val,
            stratify=temp[label_col],
            random_state=seed + 1,
        )
        rel_cal = cal_fraction / (cal_fraction + test_fraction)
        cal, test = train_test_split(
            remain,
            test_size=1.0 - rel_cal,
            stratify=remain[label_col],
            random_state=seed + 2,
        )
        return {k: v.reset_index(drop=True) for k, v in {"train": train, "val": val, "cal": cal, "test": test}.items()}
    if protocol in {"temporal_within_file", "order_preserving_within_file"}:
        buckets: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "cal": [], "test": []}
        for _, group in data.groupby("__source_file__", sort=False):
            group = group.sort_values("__row_in_file__").reset_index(drop=True)
            n = len(group)
            train_end = int(round(train_fraction * n))
            val_end = train_end + int(round(val_fraction * n))
            cal_end = val_end + int(round(cal_fraction * n))
            train_end = max(1, min(train_end, n))
            val_end = max(train_end + 1, min(val_end, n)) if n >= 2 else train_end
            cal_end = max(val_end + 1, min(cal_end, n)) if n >= 3 else val_end
            buckets["train"].append(group.iloc[:train_end])
            buckets["val"].append(group.iloc[train_end:val_end])
            buckets["cal"].append(group.iloc[val_end:cal_end])
            buckets["test"].append(group.iloc[cal_end:])
        return {name: pd.concat(parts, ignore_index=True).reset_index(drop=True) for name, parts in buckets.items()}
    raise ValueError(f"Unknown protocol: {protocol}")


def split_calibration_four_ways(
    cal_df: pd.DataFrame,
    protocol: str,
    label_col: str,
    seed: int,
    fractions: Sequence[float] = (0.40, 0.20, 0.20, 0.20),
) -> Dict[str, pd.DataFrame]:
    names = ["meta_train", "model_select", "temp_cal", "conformal_cal"]
    if len(fractions) != 4 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("Four calibration fractions summing to 1 are required.")
    if protocol == "iid_stratified":
        remaining = cal_df.copy()
        result: Dict[str, pd.DataFrame] = {}
        consumed = 0.0
        for i, (name, frac) in enumerate(zip(names[:-1], fractions[:-1])):
            relative = frac / (1.0 - consumed)
            part, remaining = train_test_split(
                remaining,
                train_size=relative,
                stratify=remaining[label_col],
                random_state=seed + 101 + i,
            )
            result[name] = part.reset_index(drop=True)
            consumed += frac
        result[names[-1]] = remaining.reset_index(drop=True)
        return result

    # Preserve order within each original source file for the chronological case.
    buckets = {name: [] for name in names}
    boundaries = np.cumsum([0.0] + list(fractions))
    for _, group in cal_df.groupby("__source_file__", sort=False):
        group = group.sort_values("__row_in_file__").reset_index(drop=True)
        n = len(group)
        cuts = [int(round(x * n)) for x in boundaries]
        cuts[0], cuts[-1] = 0, n
        for i, name in enumerate(names):
            buckets[name].append(group.iloc[cuts[i]:cuts[i + 1]])
    return {name: pd.concat(parts, ignore_index=True).reset_index(drop=True) for name, parts in buckets.items()}


@dataclass
class RunArtifacts:
    run_dir: Path
    metadata: Dict[str, object]
    label_encoder: object
    imputer: object
    variance_filter: object
    scaler: object
    input_features: List[str]
    numeric_features: List[str]
    text_features: List[str]
    xgb_model: object
    adapter_dir: Path


def load_run_artifacts(run_dir: str | Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    required = [
        "run_metadata.json", "label_encoder.joblib", "imputer.joblib",
        "variance_filter.joblib", "scaler.joblib", "input_features.joblib",
        "numeric_features.joblib", "text_features.joblib", "xgboost_model.joblib",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {run_dir}: {missing}")
    adapter_dir = run_dir / "slm_adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing SLM adapter in {adapter_dir}")
    xgb_model = joblib.load(run_dir / "xgboost_model.joblib")
    # Saved boosters were trained on GPU, while re-analysis arrays are NumPy
    # CPU arrays. For deterministic, warning-free inference, use the CPU
    # predictor. This changes the execution device, not the learned model.
    try:
        xgb_model.set_params(device="cpu")
    except Exception:
        pass
    try:
        xgb_model.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass

    return RunArtifacts(
        run_dir=run_dir,
        metadata=json.loads((run_dir / "run_metadata.json").read_text()),
        label_encoder=joblib.load(run_dir / "label_encoder.joblib"),
        imputer=joblib.load(run_dir / "imputer.joblib"),
        variance_filter=joblib.load(run_dir / "variance_filter.joblib"),
        scaler=joblib.load(run_dir / "scaler.joblib"),
        input_features=list(joblib.load(run_dir / "input_features.joblib")),
        numeric_features=list(joblib.load(run_dir / "numeric_features.joblib")),
        text_features=list(joblib.load(run_dir / "text_features.joblib")),
        xgb_model=xgb_model,
        adapter_dir=adapter_dir,
    )


def discover_runs(models_root: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(models_root)
    for metadata_path in root.glob("*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        rows.append({
            "run_dir": str(metadata_path.parent),
            "protocol": metadata.get("protocol"),
            "backbone": metadata.get("backbone"),
            "seed": int(metadata.get("seed")),
            "checkpoint": metadata.get("model_checkpoint"),
        })
    if not rows:
        raise FileNotFoundError(f"No run_metadata.json under {root}")
    return pd.DataFrame(rows).sort_values(["protocol", "backbone", "seed"]).reset_index(drop=True)


def align_frame_to_input_features(frame: pd.DataFrame, input_features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.DataFrame(index=frame.index)
    rows = []
    for feature in input_features:
        if feature in frame.columns:
            aligned[feature] = pd.to_numeric(frame[feature], errors="coerce")
            rows.append({"feature": feature, "status": "present"})
        else:
            aligned[feature] = np.nan
            rows.append({"feature": feature, "status": "missing_imputed"})
    return aligned, pd.DataFrame(rows)


def transform_with_saved_preprocessor(frame: pd.DataFrame, artifacts: RunArtifacts) -> Dict[str, np.ndarray]:
    aligned, _ = align_frame_to_input_features(frame, artifacts.input_features)
    X = aligned.to_numpy(dtype=np.float64)
    X_imp = artifacts.imputer.transform(X)
    X_var = artifacts.variance_filter.transform(X_imp)
    support = artifacts.variance_filter.get_support()
    variance_features = [f for f, keep in zip(artifacts.input_features, support) if keep]
    index = {name: i for i, name in enumerate(variance_features)}
    missing_selected = [f for f in artifacts.numeric_features if f not in index]
    if missing_selected:
        raise ValueError(f"Saved selected features missing after variance filter: {missing_selected}")
    positions = [index[f] for f in artifacts.numeric_features]
    X_num = X_var[:, positions].astype(np.float32)
    X_scaled = artifacts.scaler.transform(X_num).astype(np.float32)
    text_positions = [artifacts.numeric_features.index(f) for f in artifacts.text_features]
    return {
        "raw_numeric": X_num,
        "scaled_numeric": X_scaled,
        "scaled_text": X_scaled[:, text_positions],
    }


def encode_labels(frame: pd.DataFrame, label_col: str, artifacts: RunArtifacts) -> np.ndarray:
    return artifacts.label_encoder.transform(frame[label_col]).astype(np.int64)


class PromptDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
    def __len__(self) -> int:
        return len(self.texts)
    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class PromptCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length,
            pad_to_multiple_of=8, return_tensors="pt",
        )


def serialize_prompt(
    feature_names: Sequence[str],
    values: Sequence[float],
    mode: str = "original",
    seed: int = 0,
) -> str:
    names = list(feature_names)
    vals = np.asarray(values, dtype=float).copy()
    rng = np.random.default_rng(seed)
    if mode == "original":
        pairs = list(zip(names, vals))
    elif mode == "anonymous_names":
        pairs = [(f"feature_{i+1}", value) for i, value in enumerate(vals)]
    elif mode == "permuted_names":
        permuted = list(np.asarray(names)[rng.permutation(len(names))])
        pairs = list(zip(permuted, vals))
    elif mode == "values_only":
        return "classify: " + " ".join(f"{value:.4f}" for value in vals)
    elif mode == "names_only":
        return "classify: " + " ".join(names)
    elif mode == "shuffled_order":
        order = rng.permutation(len(names))
        pairs = [(names[i], vals[i]) for i in order]
    elif mode == "rounded_1dp":
        pairs = [(name, round(float(value), 1)) for name, value in zip(names, vals)]
    else:
        raise ValueError(f"Unknown serialization mode: {mode}")
    return "classify: " + " ".join(f"{name}={float(value):.4f}" for name, value in pairs)


def build_prompts(values: np.ndarray, features: Sequence[str], mode: str = "original", seed: int = 0) -> List[str]:
    return [serialize_prompt(features, row, mode=mode, seed=seed + i) for i, row in enumerate(values)]


def load_slm(artifacts: RunArtifacts, device: torch.device):
    """Load the saved tokenizer, base encoder, LoRA adapter, and trained head."""
    n_classes = len(artifacts.label_encoder.classes_)
    checkpoint = str(artifacts.metadata["model_checkpoint"])

    adapter_config_path = artifacts.adapter_dir / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text())
    modules_to_save = adapter_config.get("modules_to_save") or []
    if not any("classifier" in str(name).lower() for name in modules_to_save):
        warnings.warn(
            "The adapter configuration does not explicitly list the "
            "classification head in modules_to_save. Verify the original "
            "training export before interpreting predictions.",
            RuntimeWarning,
        )

    tokenizer = AutoTokenizer.from_pretrained(artifacts.adapter_dir)

    # The public base checkpoint has no task-specific classifier. Transformers
    # therefore reports a newly initialized head before PEFT restores the
    # trained head from modules_to_save. Hide that expected intermediate
    # warning, while retaining explicit adapter validation above.
    from transformers.utils import logging as hf_logging

    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )
    finally:
        hf_logging.set_verbosity(previous_verbosity)

    model = PeftModel.from_pretrained(base, artifacts.adapter_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict_slm_prompts(
    model,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int = 320,
) -> np.ndarray:
    loader = DataLoader(PromptDataset(prompts), batch_size=batch_size, shuffle=False, collate_fn=PromptCollator(tokenizer, max_length))
    outputs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_branches(
    frame: pd.DataFrame,
    artifacts: RunArtifacts,
    device: torch.device,
    batch_size: int,
    max_length: int = 320,
    serialization_mode: str = "original",
    serialization_seed: int = 0,
    loaded_slm=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], object]:
    arrays = transform_with_saved_preprocessor(frame, artifacts)
    n_classes = len(artifacts.label_encoder.classes_)
    p_num_raw = artifacts.xgb_model.predict_proba(arrays["raw_numeric"])
    p_num = align_model_probabilities(
        artifacts.xgb_model,
        p_num_raw,
        n_classes,
    )
    if loaded_slm is None:
        tokenizer, model = load_slm(artifacts, device)
    else:
        tokenizer, model = loaded_slm
    prompts = build_prompts(arrays["scaled_text"], artifacts.text_features, serialization_mode, serialization_seed)
    p_slm = validate_probability_matrix(
        predict_slm_prompts(
            model,
            tokenizer,
            prompts,
            device,
            batch_size,
            max_length,
        ),
        n_samples=len(frame),
        n_classes=n_classes,
        name="SLM probabilities",
    )
    return p_num, p_slm, arrays, (tokenizer, model)


def safe_probs(probs: np.ndarray) -> np.ndarray:
    """Clip and row-normalize a two-dimensional probability matrix."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(
            f"Expected a 2-D probability matrix, received shape {p.shape}."
        )
    if not np.isfinite(p).all():
        raise ValueError("Probability matrix contains NaN or infinite values.")
    p = np.clip(p, EPS, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("At least one probability row has a non-positive sum.")
    return p / row_sums


def align_model_probabilities(
    model,
    probabilities: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Expand model probabilities to the fixed source label space.

    Scikit-learn classifiers emit one column per class observed during fit.
    Under the order-preserving protocol, a meta-training partition can omit
    source classes. This helper maps ``model.classes_`` back to the complete
    ``0..n_classes-1`` label space, assigns a numerical floor to absent
    classes, and renormalizes each row.
    """
    raw = safe_probs(probabilities)
    model_classes = getattr(model, "classes_", None)

    if model_classes is None:
        if raw.shape[1] != n_classes:
            raise ValueError(
                "The classifier does not expose classes_, and its probability "
                f"matrix has {raw.shape[1]} columns instead of {n_classes}."
            )
        return raw

    classes = np.asarray(model_classes)
    if classes.ndim != 1 or len(classes) != raw.shape[1]:
        raise ValueError(
            "Classifier classes_ and predict_proba output are inconsistent: "
            f"{len(classes)} classes versus {raw.shape[1]} columns."
        )

    try:
        classes = classes.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Classifier classes must be integer source-label IDs: {classes!r}."
        ) from exc

    if np.any(classes < 0) or np.any(classes >= n_classes):
        raise ValueError(
            f"Classifier classes {classes.tolist()} fall outside "
            f"the source label space [0, {n_classes - 1}]."
        )
    if len(np.unique(classes)) != len(classes):
        raise ValueError(f"Classifier classes contain duplicates: {classes.tolist()}.")

    full = np.full((raw.shape[0], n_classes), EPS, dtype=np.float64)
    full[:, classes] = raw
    return safe_probs(full)


def validate_probability_matrix(
    probabilities: np.ndarray,
    n_samples: int,
    n_classes: int,
    name: str,
) -> np.ndarray:
    """Validate and normalize probabilities before metric computation."""
    p = safe_probs(probabilities)
    expected = (int(n_samples), int(n_classes))
    if p.shape != expected:
        raise ValueError(
            f"{name} has shape {p.shape}; expected {expected}. "
            "A meta-classifier may have omitted classes from predict_proba."
        )
    return p


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((safe_probs(probs) - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    p = safe_probs(probs)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def metrics_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, received {y_true.shape}.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty partition.")
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError(
            "y_true contains labels outside the fixed source label space: "
            f"{np.unique(y_true).tolist()}."
        )

    p = validate_probability_matrix(
        probs,
        n_samples=len(y_true),
        n_classes=n_classes,
        name="Metric probability matrix",
    )
    pred = p.argmax(axis=1)
    all_labels = np.arange(n_classes)
    present = np.unique(y_true)

    # For partitions containing only a subset of source classes, balanced
    # accuracy is the macro recall over classes actually represented in y_true.
    balanced_present = recall_score(
        y_true,
        pred,
        labels=present,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1_present": float(
            f1_score(
                y_true,
                pred,
                labels=present,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_all": float(
            f1_score(
                y_true,
                pred,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(y_true, pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_present),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "macro_precision_present": float(
            precision_score(
                y_true,
                pred,
                labels=present,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall_present": float(balanced_present),
        "nll": float(log_loss(y_true, p, labels=all_labels)),
        "brier": multiclass_brier(y_true, p, n_classes),
        "ece": expected_calibration_error(y_true, p),
        "present_class_count": int(len(present)),
        "label_space_coverage": float(len(present) / n_classes),
    }


def candidate_score(metrics: Mapping[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics["macro_f1_present"]),
        float(metrics["mcc"]),
        -float(metrics["nll"]),
    )


def log_probability_features(p_num: np.ndarray, p_slm: np.ndarray) -> np.ndarray:
    return np.concatenate([np.log(safe_probs(p_num)), np.log(safe_probs(p_slm))], axis=1)


def _fit_meta_classifier(
    features: np.ndarray,
    y: np.ndarray,
    seed: int,
):
    """Fit a probabilistic meta-classifier without failing on one-class splits."""
    y = np.asarray(y, dtype=int)
    observed = np.unique(y)
    if len(observed) == 0:
        raise ValueError("Cannot fit a meta-classifier on an empty partition.")
    if len(observed) == 1:
        warnings.warn(
            "The meta-training partition contains one class only; using a "
            "constant probabilistic control for this run.",
            RuntimeWarning,
        )
        model = DummyClassifier(strategy="prior")
    else:
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
    model.fit(features, y)
    return model


def fit_stacker(
    p_num: np.ndarray,
    p_slm: np.ndarray,
    y: np.ndarray,
    seed: int,
):
    return _fit_meta_classifier(
        log_probability_features(p_num, p_slm),
        y,
        seed,
    )


def fit_duplicate_numeric_stacker(
    p_num: np.ndarray,
    y: np.ndarray,
    seed: int,
):
    p_num = safe_probs(p_num)
    features = np.concatenate([np.log(p_num), np.log(p_num)], axis=1)
    return _fit_meta_classifier(features, y, seed)


def predict_stacker(
    model: LogisticRegression,
    p_num: np.ndarray,
    p_slm: np.ndarray,
) -> np.ndarray:
    n_classes = safe_probs(p_num).shape[1]
    raw = model.predict_proba(log_probability_features(p_num, p_slm))
    return align_model_probabilities(model, raw, n_classes)


def predict_duplicate_numeric_stacker(
    model: LogisticRegression,
    p_num: np.ndarray,
) -> np.ndarray:
    p_num = safe_probs(p_num)
    n_classes = p_num.shape[1]
    X = np.concatenate([np.log(p_num), np.log(p_num)], axis=1)
    raw = model.predict_proba(X)
    return align_model_probabilities(model, raw, n_classes)


def select_convex_alpha(p_num: np.ndarray, p_slm: np.ndarray, y: np.ndarray, n_classes: int, grid_points: int = 41) -> Tuple[float, pd.DataFrame]:
    rows = []
    for alpha in np.linspace(0.0, 1.0, grid_points):
        probs = (1.0 - alpha) * p_num + alpha * p_slm
        metrics = metrics_from_probs(y, probs, n_classes)
        rows.append({"alpha": float(alpha), **metrics})
    table = pd.DataFrame(rows)
    table = table.sort_values(["macro_f1_present", "mcc", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def build_candidate_probabilities(
    p_num: np.ndarray,
    p_slm: np.ndarray,
    alpha: float,
    stacker: LogisticRegression,
    numeric_control: LogisticRegression,
) -> Dict[str, np.ndarray]:
    return {
        "xgboost": safe_probs(p_num),
        "slm": safe_probs(p_slm),
        "fixed_average": safe_probs(0.5 * p_num + 0.5 * p_slm),
        f"convex_alpha_{alpha:.3f}": safe_probs((1.0 - alpha) * p_num + alpha * p_slm),
        "stacking": safe_probs(predict_stacker(stacker, p_num, p_slm)),
        "numeric_duplicate_stacking_control": safe_probs(predict_duplicate_numeric_stacker(numeric_control, p_num)),
    }


def select_candidate(candidates: Mapping[str, np.ndarray], y: np.ndarray, n_classes: int) -> Tuple[str, pd.DataFrame]:
    rows = []
    for name, probs in candidates.items():
        rows.append({"candidate": name, **metrics_from_probs(y, probs, n_classes)})
    table = pd.DataFrame(rows).sort_values(["macro_f1_present", "mcc", "nll"], ascending=[False, False, True]).reset_index(drop=True)
    return str(table.iloc[0]["candidate"]), table



def run_full_class_probability_self_test() -> Dict[str, object]:
    """Verify that a stacker trained on partial labels still returns all columns."""
    rng = np.random.default_rng(20260726)
    n_classes = 10
    n_train = 80
    n_eval = 17

    p_num_train = safe_probs(rng.random((n_train, n_classes)))
    p_slm_train = safe_probs(rng.random((n_train, n_classes)))
    # Deliberately omit five source classes, matching the difficult protocol.
    partial_classes = np.array([0, 2, 4, 6, 8], dtype=int)
    y_train = np.resize(partial_classes, n_train)

    stacker = fit_stacker(p_num_train, p_slm_train, y_train, seed=42)
    numeric_control = fit_duplicate_numeric_stacker(
        p_num_train,
        y_train,
        seed=42,
    )

    p_num_eval = safe_probs(rng.random((n_eval, n_classes)))
    p_slm_eval = safe_probs(rng.random((n_eval, n_classes)))
    p_stack = predict_stacker(stacker, p_num_eval, p_slm_eval)
    p_control = predict_duplicate_numeric_stacker(
        numeric_control,
        p_num_eval,
    )

    assert p_stack.shape == (n_eval, n_classes)
    assert p_control.shape == (n_eval, n_classes)
    assert np.allclose(p_stack.sum(axis=1), 1.0)
    assert np.allclose(p_control.sum(axis=1), 1.0)

    # Exercise the exact metric path that previously failed.
    y_eval = np.resize(np.arange(n_classes), n_eval)
    stack_metrics = metrics_from_probs(y_eval, p_stack, n_classes)
    control_metrics = metrics_from_probs(y_eval, p_control, n_classes)

    return {
        "utils_version": UTILS_VERSION,
        "stacker_observed_classes": stacker.classes_.tolist(),
        "stacker_probability_shape": tuple(p_stack.shape),
        "control_probability_shape": tuple(p_control.shape),
        "stacker_nll": stack_metrics["nll"],
        "control_nll": control_metrics["nll"],
        "status": "passed",
    }

def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.log(safe_probs(probs))


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits_from_probs(probs) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    def objective(log_t: float) -> float:
        t = math.exp(float(log_t))
        calibrated = temperature_scale_probs(probs, t)
        return float(log_loss(y, calibrated, labels=np.arange(calibrated.shape[1])))
    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def aps_scores(probs: np.ndarray, y: np.ndarray, raps_lambda: float = 0.0, k_reg: int = 0) -> np.ndarray:
    p = safe_probs(probs)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1)
    inverse_rank = np.empty_like(order)
    inverse_rank[np.arange(len(p))[:, None], order] = np.arange(p.shape[1])[None, :]
    ranks = inverse_rank[np.arange(len(p)), y]
    base = cumulative[np.arange(len(p)), ranks]
    penalty = raps_lambda * np.maximum((ranks + 1) - k_reg, 0)
    return base + penalty


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, level, interpolation="higher"))


def prediction_sets(probs: np.ndarray, qhat: float, raps_lambda: float = 0.0, k_reg: int = 0) -> List[np.ndarray]:
    p = safe_probs(probs)
    sets: List[np.ndarray] = []
    for row in p:
        order = np.argsort(-row)
        cumulative = 0.0
        selected = []
        for rank, cls in enumerate(order, start=1):
            cumulative += float(row[cls])
            score = cumulative + raps_lambda * max(rank - k_reg, 0)
            selected.append(int(cls))
            if score >= qhat:
                break
        sets.append(np.asarray(selected, dtype=np.int64))
    return sets


def summarize_prediction_sets(sets: Sequence[np.ndarray], y: np.ndarray, abstain_size: int = 2) -> Dict[str, float]:
    sizes = np.asarray([len(s) for s in sets])
    covered = np.asarray([int(int(label) in set(map(int, pred_set))) for pred_set, label in zip(sets, y)])
    return {
        "coverage": float(covered.mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "ambiguity_rate": float((sizes == 2).mean()),
        "abstention_rate": float((sizes > abstain_size).mean()),
    }


def evaluate_binary_attack_detection(raw_is_attack: np.ndarray, source_probs: np.ndarray, source_classes: Sequence[str]) -> Dict[str, float]:
    classes = list(source_classes)
    benign_index = classes.index("BENIGN")
    pred_attack = source_probs.argmax(axis=1) != benign_index
    y = np.asarray(raw_is_attack, dtype=bool)
    pred = np.asarray(pred_attack, dtype=bool)
    return {
        "binary_accuracy": float(accuracy_score(y, pred)),
        "binary_f1": float(f1_score(y, pred, zero_division=0)),
        "binary_mcc": float(matthews_corrcoef(y, pred)),
        "attack_recall": float(recall_score(y, pred, zero_division=0)),
        "attack_precision": float(precision_score(y, pred, zero_division=0)),
    }


def clear_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
