from typing import List

import lightgbm
import numpy as np
from pathlib import Path


class LGBMCVModel:
    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.models: List[lightgbm.Booster] = []

        self.__load_models()

    def __load_models(self):
        for model in (self.run_path / "models").iterdir():
            model = lightgbm.Booster(model_file=model)
            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(X))
        return np.array(predictions) / len(self.models)
