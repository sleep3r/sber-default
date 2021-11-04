import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline

from config import MLConfig


class DefaultTransformer:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame = None):
        self.cfg = cfg
        self.X = X
        self.X_test = X_test

    def _drop_duplicates(self) -> pd.DataFrame:
        subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + \
                 [self.cfg.dataset.target_name]
        x = self.X.copy().drop_duplicates(subset, keep="last")
        return x

    def _get_column_transformer(self) -> ColumnTransformer:
        cont_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
        )

        column_transformer = ColumnTransformer([
            ('ohe', OneHotEncoder(handle_unknown='ignore'), self.cfg.preprocessing.cat_features),
            ("num", cont_transformer, self.cfg.preprocessing.cont_features),
            ('other', 'passthrough', self.cfg.preprocessing.other_features)
        ])
        return column_transformer

    def transform(self):
        if self.cfg.preprocessing.drop_duplicates:
            x = self._drop_duplicates()
        else:
            x = self.X.copy().drop(self.cfg.dataset.target_name, axis=1)
        y = self.X[self.cfg.dataset.target_name].copy()

        x_test = self.X_test.copy() if self.X_test is not None else None

        column_transformer = self._get_column_transformer()
        x = column_transformer.fit_transform(x)
        return {"train": (x, y), "test": x_test}
