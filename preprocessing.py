from typing import List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer

from config import MLConfig, object_from_dict


def get_column_names_from_ColumnTransformer(column_transformer: ColumnTransformer) -> List[str]:
    col_name = []

    for transformer_in_columns in column_transformer.transformers_[:-1]:
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, OneHotEncoder):
                names = list(transformer.get_feature_names(raw_col_name))
            elif isinstance(transformer, SimpleImputer) or isinstance(transformer, KNNImputer) \
                    and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]
                names = raw_col_name + missing_indicators
            else:
                names = list(transformer.get_feature_names())
        except AttributeError as error:
            names = raw_col_name
        col_name.extend(names)
    return col_name


class DefaultTransformer:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame = None):
        self.cfg = cfg
        self.X = X.copy()
        self.X_test = X_test.copy()

    def _drop_duplicates(self) -> None:
        subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + \
                 [self.cfg.dataset.target_name]
        self.X = self.X.drop_duplicates(subset, keep="last")

    def _group_duplicates(self) -> None:
        subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + \
                 [self.cfg.dataset.target_name]
        self.X["duplicates_group"] = self.X.fillna(-1).groupby(subset).ngroup()

    def _make_df(self, x: np.ndarray, transformer: ColumnTransformer) -> pd.DataFrame:
        x = pd.DataFrame(x)
        x.columns = get_column_names_from_ColumnTransformer(transformer)
        return x

    def _fit_transform(self, x, x_test, column_transformer) -> (pd.DataFrame, pd.DataFrame):
        x = column_transformer.fit_transform(x)
        x = self._make_df(x, column_transformer)

        x_test = column_transformer.fit_transform(x_test)
        x_test = self._make_df(x_test, column_transformer)
        return x, x_test

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
            ('other', imputer, self.cfg.preprocessing.other_features)
        ])
        return column_transformer

    def transform(self) -> dict:
        if self.cfg.preprocessing.replace_inf is not None:
            self.X = self.X.replace({np.inf: self.cfg.preprocessing.replace_inf}).copy()
            self.X_test = self.X_test.replace({np.inf: self.cfg.preprocessing.replace_inf}).copy()

        if self.cfg.preprocessing.duplicates == "drop":
            self._drop_duplicates()
        elif self.cfg.preprocessing.duplicates == "group":
            self._group_duplicates()

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
