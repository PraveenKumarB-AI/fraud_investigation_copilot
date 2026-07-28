"""
models/train_baseline.py — MODULE 4: XGBoost baseline (no graph)
Trains XGBoost on the same 165 features used by the GNN, with the identical
time-based split (train: steps 1-27, test: steps 35-49), to test whether
the graph structure in Module 3 is actually earning its keep. Saves the
trained model — this becomes the scoring model used in later modules,
since it beat GraphSAGE on every metric.
Run:  python -m models.train_baseline
"""

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)

PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "models/checkpoints"
TRAIN_MAX_STEP = 27
VAL_MAX_STEP = 34


def main():
    print("1. Loading processed data...")
    merged = pd.read_csv(f"{PROCESSED_DIR}/elliptic_merged.csv")
    feature_cols = [c for c in merged.columns if c.startswith("feat_")]

    label_map = {"1": 1, "2": 0, "unknown": -1}
    labels = merged["class"].astype(str).map(label_map)
    merged = merged.assign(label=labels)

    labeled = merged[merged["label"] != -1].copy()
    print(f"   {len(labeled)} labeled transactions")

    print("\n2. Time-based split (matching Module 3's boundaries exactly)...")
    train = labeled[labeled["time_step"] <= TRAIN_MAX_STEP]
    test = labeled[labeled["time_step"] > VAL_MAX_STEP]
    print(f"   Train: {len(train)} (steps 1-{TRAIN_MAX_STEP})")
    print(f"   Test:  {len(test)} (steps {VAL_MAX_STEP+1}-49)")

    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    print("\n3. Training XGBoost (class-weighted for the same imbalance the GNN faced)...")
    n_illicit = (y_train == 1).sum()
    n_licit = (y_train == 0).sum()
    scale_pos_weight = n_licit / n_illicit
    print(f"   Train — illicit: {n_illicit}, licit: {n_licit}, scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\n4. Evaluating on the held-out test set (steps 35-49)...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_pr = average_precision_score(y_test, probs)

    print(f"\n{'='*55}")
    print(f"XGBoost (no graph) — Illicit precision: {precision*100:.1f}%")
    print(f"XGBoost (no graph) — Illicit recall:    {recall*100:.1f}%")
    print(f"XGBoost (no graph) — Illicit F1:        {f1*100:.1f}%")
    print(f"XGBoost (no graph) — AUC-PR:            {auc_pr*100:.1f}%")
    print(f"{'='*55}")
    print(f"\nConfusion matrix [rows=actual, cols=predicted], order [licit, illicit]:")
    print(confusion_matrix(y_test, preds))

    print("\n5. Saving model and the feature column order (needed for scoring new transactions)...")
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(f"{CHECKPOINT_DIR}/xgb_baseline.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"   Saved {CHECKPOINT_DIR}/xgb_baseline.pkl")


if __name__ == "__main__":
    main()
