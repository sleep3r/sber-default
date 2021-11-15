from typing import Dict
from pathlib import Path

import lightgbm
import numpy as np


class LGBMCVModel:
    def __init__(self, run_path: str):
        self.run_path = Path(run_path)
        self.models: Dict[str, lightgbm.Booster] = {}

        self.__load_models()

    def __load_models(self):
        for model_path in (self.run_path).glob("*.model"):
            model = lightgbm.Booster(model_file=str(model_path))
            model.params["objective"] = "binary"
            self.models[model_path.stem] = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for name, model in self.models.items():
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)
