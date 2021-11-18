from typing import Dict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from config import MLConfig, object_from_dict
from utils.scikitlearn import get_column_names_from_ColumnTransformer


class DefaultTransformer:
    """
    Base feature transformer class.

    Attributes:
        config (MLConfig): configuration object;
        X (pd.DataFrame): train input data;
        X_test (pd.DataFrame): test input data.
    """

    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame = None):
        self.cfg = cfg
        self.X = X.copy()
        self.X_test = X_test.copy()

    def make_df(self, x: np.ndarray, transformer: ColumnTransformer) -> pd.DataFrame:
        """
        Makes dataframe from transformer output.

        Args:
            x (np.ndarray): transformer output;
            transformer (ColumnTransformer): fitted ColumnTransformer.
        Returns:
            pd.DataFrame: transformed data dataframe.
        """
        x = pd.DataFrame(x).astype(float)
        x.columns = get_column_names_from_ColumnTransformer(transformer)
        return x

    def get_column_transformer(self) -> ColumnTransformer:
        """Loads configurated feature transformers."""
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

    def fit_transform(
            self,
            x: pd.DataFrame,
            x_test: pd.DataFrame,
            column_transformer: ColumnTransformer
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Base method with fit_transform.

        Args:
            x (pd.DataFrame): train input data;
            x_test (pd.DataFrame): test input data;
            column_transformer (ColumnTransformer): fitted ColumnTransformer.
        Returns:
            (pd.DataFrame, pd.DataFrame): transformed train and test data.
        """
        column_transformer.fit(x)
        x = column_transformer.transform(x)
        x = self.make_df(x, column_transformer)

        x_test = column_transformer.transform(x_test)
        x_test = self.make_df(x_test, column_transformer)
        return x, x_test

    def transform(self) -> Dict[str, pd.DataFrame]:
        """Base transform method."""
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

        column_transformer = self.get_column_transformer()
        x, x_test = self.fit_transform(x, x_test, column_transformer)
        return {"train": (x, y), "test": x_test}
