from typing import Dict
from pathlib import Path

import lightgbm as lgb
import numpy as np


class LGBMCVModel:
    """
    Base class for cross-validated model lgbm.

    Attributes:
        run_path (Path): path to the run directory;
        models (Dict[str, lgb.Booster]): dictionary of models.
    """

    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.models: Dict[str, lgb.Booster] = {}

        self.load_models()

    def load_models(self) -> None:
        """Loads all fold models from run path"""
        for model_path in (self.run_path).glob("*.model"):
            model = lgb.Booster(model_file=str(model_path))
            model.params["objective"] = "binary"
            self.models[model_path.stem] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Base predict method.

        Args:
            X (np.ndarray): features array.
        Returns:
            np.ndarray: predictions probas.
        """
        predictions = []
        for name, model in self.models.items():
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)
