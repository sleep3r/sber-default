import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from config import MLConfig, object_from_dict
from utils.scikitlearn import get_column_names_from_ColumnTransformer


class DefaultTransformer:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame = None):
        self.cfg = cfg
        self.X = X.copy()
        self.X_test = X_test.copy()

    def _make_df(self, x: np.ndarray, transformer: ColumnTransformer) -> pd.DataFrame:
        x = pd.DataFrame(x).astype(float)
        x.columns = get_column_names_from_ColumnTransformer(transformer)
        return x

    def _get_column_transformer(self) -> ColumnTransformer:
        scaler = object_from_dict(self.cfg.preprocessing.scaler)
        imputer = object_from_dict(self.cfg.preprocessing.imputer)
        encoder = object_from_dict(self.cfg.preprocessing.encoder)
        normalizer = object_from_dict(self.cfg.preprocessing.normalizer)

        cont_transformer = Pipeline(
            steps=[("imputer", imputer), ("scaler", scaler)]
        )
        skewed_transformer = Pipeline(
            steps=[("imputer", imputer), ("normalizer", normalizer), ("scaler", scaler)]
        )

        column_transformer = ColumnTransformer([
            ('ohe', encoder, self.cfg.preprocessing.cat_features),
            ("num", cont_transformer, self.cfg.preprocessing.cont_features),
            ("skewed", skewed_transformer, self.cfg.preprocessing.skewed_features),
            ('other', 'passthrough', self.cfg.preprocessing.other_features)
        ])
        return column_transformer

    def _fit_transform(self, x, x_test, column_transformer) -> (pd.DataFrame, pd.DataFrame):
        x.to_csv("a.csv")
        x = column_transformer.fit_transform(x)
        x = self._make_df(x, column_transformer)

        x_test = column_transformer.fit_transform(x_test)
        x_test = self._make_df(x_test, column_transformer)
        return x, x_test

    def transform(self) -> dict:
        if self.cfg.preprocessing.replace_inf is not None:
            repl = {
                np.inf: self.cfg.preprocessing.replace_inf,
                -np.inf: self.cfg.preprocessing.replace_inf
            }
            self.X = self.X.replace(repl).copy()
            self.X_test = self.X_test.replace(repl).copy()

        if self.cfg.preprocessing.select_rows is not None:
            self.X = self.X[self.X[self.cfg.preprocessing.select_rows]].copy()
            self.X_test = self.X_test[self.X_test[self.cfg.preprocessing.select_rows]].copy()

        if self.cfg.preprocessing.process_na == "drop":
            self.X = self.X.dropna()
        elif self.cfg.preprocessing.process_na == "keep":
            self.X = self.X[self.X.isnull().any(1)].copy()

        x = self.X.drop(self.cfg.dataset.target_name, axis=1)
        y = self.X[self.cfg.dataset.target_name].copy()

        x_test = self.X_test.copy() if self.X_test is not None else None

        column_transformer = self._get_column_transformer()
        x, x_test = self._fit_transform(x, x_test, column_transformer)
        return {"train": (x, y), "test": x_test}
