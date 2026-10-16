# SemCons-IDS — CRiSIS 2026 Validated Results

## Experimental setup

- Dataset: CICIDS2017
- Protocols: IID stratified and order-preserving within-file
- Semantic backbones: MiniLM and ModernBERT
- Seeds: 42, 123, 2026
- Total evaluated configurations: 12

## Validation

The original preprocessing and test partitions were
reconstructed from the comparative v8 notebook.

The saved XGBoost models were re-evaluated on the
reconstructed test sets. All 12 configurations passed
the implemented consistency checks against the
historical Colab metrics.

## Main results

Mean Macro-F1 gains over XGBoost:

- IID / MiniLM: -0.097 percentage points
- IID / ModernBERT: -0.036 percentage points
- Order-preserving / MiniLM: +18.466 percentage points
- Order-preserving / ModernBERT: +17.973 percentage points

Under the order-preserving protocol, only five of
the ten modeled classes appear in the test partition.
The reported primary Macro-F1 averages over these
five present classes.

## Class-level findings

Under order-preserving evaluation, the largest
improvements concern Bot and DoS_GoldenEye.

Semantic fusion also introduces some additional
errors on benign traffic.

## Limitations

- Order-preserving within-file evaluation is not
  equivalent to cross-dataset generalization.
- Only three random seeds were evaluated.
- Corrected and introduced errors are aggregated
  across runs, not counted as unique network flows.
- Matching historical metrics is a consistency check,
  not independent verification of every original sample.
- Raw datasets, saved predictions, and model weights
  are not distributed.

## Reproduction

See scripts/compare_all_runs.py.

The analysis requires access to the original
CICIDS2017 files and saved model artifacts.
