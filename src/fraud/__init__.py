"""Fraud detection pipeline (Task 1 of X-DRA).

Modules:
    config          paths, split fractions, business assumptions, column lists
    preprocess      loading, validation, chronological split, feature building
    models          model factories, Optuna tuning, resource profiling
    calibration     probability calibration and reliability diagnostics
    evaluate        PR-AUC, FPR-constrained threshold, operating-point reporting
    expected_value  calibrated probabilities -> shared-schema candidate table
    train           end-to-end pipeline and CLI (``python -m src.fraud.train``)
"""
