from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split

from config import MLConfig


def get_train_folds(cfg: MLConfig, X_preprocessed: pd.DataFrame, y: pd.Series) -> list:
    if cfg.validation.mode == "holdout":
        X_train, X_val, y_train, y_val = train_test_split(
            X_preprocessed, y,
            stratify=y, test_size=cfg.validation.params.test_size
        )
        folds = [[(X_train, y_train), (X_val, y_val)]]  # one fold
        return folds
    elif cfg.validation.mode == "CV":
        pass
    else:
        raise NotImplementedError("WTF validation mode", cfg.validation.mode)


def validate(
        meta: dict, preds: np.ndarray, y_val: np.ndarray, fold_id: int
) -> Dict[str, float]:
    metrics = {}

    metrics["clf_report"] = classification_report(y_val, preds)
    metrics["accuracy"] = accuracy_score(y_val, preds)
    metrics["roc_auc"] = roc_auc_score(y_val, preds)
    metrics["precision"] = precision_score(y_val, preds)
    metrics["recall"] = recall_score(y_val, preds)
    metrics["F1"] = f1_score(y_val, preds)
    return metrics
