from typing import Dict

import pandas as pd

from .config import MLConfig


class DefaultDataset:
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.__load()

    def __load(self) -> None:
        self.train_df = pd.read_csv(self.cfg.dataset.train_path, sep=";")
        self.y = self.train_df[self.cfg.dataset.target_name]
        self.train_df = self.train_df.drop(self.cfg.dataset.target_name, axis=1)

        self.test_df = pd.read_csv(self.cfg.dataset.test_path, sep=";")

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        return {"train": (self.train_df, self.y), "test": self.test_df}
