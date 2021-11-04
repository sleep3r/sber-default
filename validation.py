import pandas as pd
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
