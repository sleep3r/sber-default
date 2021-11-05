import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import MLConfig, object_from_dict


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

    def transform(self):
        if self.cfg.preprocessing.drop_duplicates:
            x = self._drop_duplicates()
        else:
            x = self.X.copy().drop(self.cfg.dataset.target_name, axis=1)
        y = self.X[self.cfg.dataset.target_name].copy()

        x_test = self.X_test.copy() if self.X_test is not None else None

        if self.cfg.preprocessing.has_na:
            x["has_na"] = x[self.cfg.preprocessing.has_na].isna()

        column_transformer = self._get_column_transformer()
        x = column_transformer.fit_transform(x)
        return {"train": (x, y), "test": x_test}
