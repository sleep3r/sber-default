import numpy as np
import pandas as pd

from config import MLConfig


class DefaultGenerator:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame):
        self.cfg = cfg
        self.X = X.copy()
        self.X_test = X_test.copy()

    def _drop_duplicates(self) -> pd.DataFrame:
        subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + \
                 [self.cfg.dataset.target_name]
        return self.X.drop_duplicates(subset, keep="last")

    def _group_duplicates(self) -> pd.DataFrame:
        subset = [*set(self.X.columns) - set(self.cfg.preprocessing.drop_features)] + \
                 [self.cfg.dataset.target_name]
        self.X["duplicates_group"] = self.X.fillna(-1).groupby(subset).ngroup()
        return self.X

    def _generate(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.cfg.features.generation.has_na:
            X["has_na"] = 0
            X.loc[X.isnull().any(1), "has_na"] = 1

        if self.cfg.features.generation.duplicates == "drop":
            X = self._drop_duplicates()
        elif self.cfg.features.generation.duplicates == "group":
            X = self._group_duplicates()

        TL = X.ab_long_term_liabilities + X.ab_other_borrowings + X.ab_short_term_borrowing
        TA = X.ab_own_capital + X.ab_borrowed_capital
        STD = X.ab_short_term_borrowing
        STFD = X.ab_short_term_borrowing
        CA = X.ab_mobile_current_assets
        FCA = X.ab_mobile_current_assets - X.ab_inventory
        GY = X.ar_sale_profit

        X['r_1_a'] = X.ar_revenue / (X.ab_accounts_receivable / 12)
        X['r_2_a'] = X.ar_sale_cost / (X.ab_inventory / 12)
        X['r_3_a'] = X.ar_selling_expenses / (X.ar_total_expenses / 12)
        X['r_4_a'] = X.ar_revenue / TA - TL
        X['r_5_a'] = X.ar_revenue / (X.ab_immobilized_assets / 12)
        X['r_6_a'] = X.ar_revenue / (X.ab_mobile_current_assets + X.ab_cash_and_securities)
        X['r_7_a'] = X.ar_sale_profit / X.ar_revenue
        X['r_8_a'] = X.ar_profit_before_tax / X.ar_revenue
        X['r_9_a'] = X.ar_net_profit / X.ar_revenue

        X['r_10_a'] = X.ab_short_term_borrowing / \
                      (X.ab_short_term_borrowing + X.ab_accounts_payable + X.ab_other_borrowings)
        X['r_11_a'] = X.ab_accounts_payable / \
                      (X.ab_short_term_borrowing + X.ab_accounts_payable + X.ab_other_borrowings)
        X['r_12_a'] = X.ab_inventory / X.ar_revenue
        X['r_13_a'] = X.ab_long_term_liabilities / X.ar_revenue
        X['r_15_a'] = X.ar_taxes / X.ar_revenue

        X['r_16_a'] = X.ab_inventory / X.ab_borrowed_capital
        X['r_17_a'] = X.ab_inventory / X.ab_mobile_current_assets
        X['r_18_a'] = X.ab_inventory / X.ab_accounts_payable

        X['r_19_a'] = X.ab_accounts_receivable / \
                      (X.ab_cash_and_securities + X.ab_accounts_receivable)
        X['r_20_a'] = X.ab_cash_and_securities / X.ab_borrowed_capital
        X['r_21_a'] = X.ab_cash_and_securities / X.ab_short_term_borrowing
        X['r_22_a'] = X.ab_cash_and_securities / \
                      (X.ab_short_term_borrowing + X.ab_accounts_payable)
        X['r_23_a'] = X.ab_cash_and_securities / \
                      (X.ab_short_term_borrowing + X.ab_accounts_payable + X.ab_other_borrowings)

        X['r_24_a'] = X.ar_profit_before_tax / X.ar_net_profit

        X['r_25_a'] = TL / TA
        X['r_26_a'] = (X.ab_accounts_receivable + X.ab_cash_and_securities) / TA
        X['r_27_a'] = CA / STD
        return X

    def generate_features(self) -> (pd.DataFrame, pd.DataFrame):
        X = self._generate(self.X)
        X_test = self._generate(self.X_test)
        return X, X_test


class DefaultSelector:
    def __init__(self, cfg: MLConfig, X: pd.DataFrame, X_test: pd.DataFrame):
        self.cfg = cfg
        self.X = X.copy()
        self.X_test = X_test.copy()

    def _select(self, X: pd.DataFrame):
        X = X.copy()
        return X[self.cfg.features.selection.selected]

    def select_features(self) -> (pd.DataFrame, pd.DataFrame):
        X = self._select(self.X)
        X_test = self._select(self.X_test)
        return X, X_test
