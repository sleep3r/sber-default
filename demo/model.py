from typing import List
from pathlib import Path

import lightgbm
import numpy as np


class LGBMCVModel:
    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.models: List[lightgbm.Booster] = []

        self.__load_models()

    def __load_models(self):
        for model_path in (self.run_path / "models").iterdir():
            model = lightgbm.Booster(model_file=str(model_path))
            self.models.append(model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)
