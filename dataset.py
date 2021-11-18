from typing import Dict

import pandas as pd

from config import MLConfig


class DefaultDataset:
    """
    Base dataset class.

    Attributes:
        cfg (MLConfig): Configuration object.
    """

    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.load()

    def load(self) -> None:
        """Loads data from cfg paths."""
        self.train_df = pd.read_csv(self.cfg.dataset.train_path, sep=";")
        self.test_df = pd.read_csv(self.cfg.dataset.test_path, sep=";")

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Returns data as dictionary."""
        return {"train": self.train_df, "test": self.test_df}
