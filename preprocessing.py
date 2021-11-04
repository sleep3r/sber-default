import pandas as pd

from config import MLConfig


class DefaultTransformer:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame = None):
        self.cfg = cfg
        self.X = X
        self.X_test = X_test

    def transform(self):
        x = self.X.copy()
        x_test = self.X_test.copy() if self.X_test else None

        if self.cfg.preprocessing.drop_duplicates:
            subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + ["default_12m"]
            x = x.drop_duplicates(subset, keep="last")

        return {"train": x, "test": x_test}
