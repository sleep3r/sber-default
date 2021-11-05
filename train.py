import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split

from config import load_config, MLConfig, object_from_dict
from dataset import DefaultDataset
from preprocessing import DefaultTransformer
from validation import get_train_folds, validate
from utils.env import collect_env
from utils.path import mkdir_or_exist


def create_workdir(cfg: MLConfig, meta: dict) -> dict:
    dirname = f"{cfg.exp_name}_{datetime.now().strftime('%d.%m/%H.%M.%S')}"
    meta["exp_dir"] = Path(cfg.work_dir) / dirname
    mkdir_or_exist(meta["exp_dir"])
    return meta


def env_collect(cfg: MLConfig, meta: dict) -> dict:
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'

    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    meta['env_info'] = env_info
    return meta


def set_random_seed(seed: int = 228):
    """
    Sets random seed.

    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def determine_exp(cfg: MLConfig, meta: dict) -> dict:
    if cfg.seed is not None:
        print(f'\nSet random seed to {cfg.seed}')
        set_random_seed(cfg.seed)

    meta['seed'] = cfg.seed
    meta['exp_name'] = cfg.exp_name
    return meta


def log_report(meta: dict, exp_dir: Path) -> None:
    print("\nClassification report best model:")
    print(meta["metrics"].pop("clf_report"))

    print("Metrics:")
    for name, value in meta["metrics"].items():
        print(f"{name} = {value}")

    with open(exp_dir / "report.json", "w") as f:
        json.dump(meta, f, indent=4)


def log_best_model(best_estimator, exp_dir: Path) -> None:
    with open(exp_dir / "best.pkl", "wb") as f:
        pickle.dump(best_estimator, f)


def make_submit(model, X_preprocessed, y, X_test_preprocessed, index: np.ndarray) -> pd.DataFrame:
    model.fit(X_preprocessed, y)

    predict = (model.predict_proba(X_test_preprocessed)[:, 1] > y.mean()).astype(int)
    answ_df = pd.DataFrame(index, columns=["id"])
    answ_df['predict'] = predict
    return answ_df


def log_dataset(X: pd.DataFrame, exp_dir: Path) -> None:
    X.to_csv(exp_dir / "data.csv", header=None, index=None)


def log_submit(submit_df: pd.DataFrame, exp_dir: Path) -> None:
    submit_df.to_csv(exp_dir / 'PD-submit.csv', index=False, sep=';')


def log_artifacts(meta: dict, best_estimator, X: pd.DataFrame, submit_df: pd.DataFrame) -> None:
    exp_dir = meta.pop("exp_dir")

    log_report(meta, exp_dir)
    log_best_model(best_estimator, exp_dir)
    log_dataset(X, exp_dir)
    log_submit(submit_df, exp_dir)


def prepare_training(cfg: MLConfig) -> dict:
    meta = dict()

    # create work_dir
    create_workdir(cfg, meta)

    # dump config
    cfg.dump(meta["exp_dir"] / "config.yml")

    # log env info
    env_collect(cfg, meta)

    # log config
    cfg.pretty_print()

    # set random seeds
    determine_exp(cfg, meta)
    return meta


def train_model(cfg: MLConfig):
    meta: dict = prepare_training(cfg)

    print("\nLoading dataset...")
    dataset = DefaultDataset(cfg)
    X, X_test = dataset.data["train"], dataset.data["test"]

    print("\nPreprocessing dataset...")
    transformer = DefaultTransformer(cfg, X, X_test)
    preprocessed = transformer.transform()
    (X_preprocessed, y), X_test_preprocessed = preprocessed["train"], preprocessed["test"]

    print("\nGenerating features...")
    # feature selection
    # pass

    print("\nTraining...")
    model = object_from_dict(cfg.model)

    X_train, X_val, y_train, y_val = train_test_split(
        X_preprocessed, y,
        stratify=y, test_size=cfg.validation.test_size, shuffle=True
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = validate(preds=preds, y_val=y_val)
    meta["metrics"] = metrics

    try:
        cv = cross_validate(
            object_from_dict(cfg.model), X_preprocessed, y,
            cv=cfg.validation.n_folds, scoring=cfg.validation.scoring
        )
        for metric in cfg.validation.scoring:
            meta["metrics"][f"CV_{cfg.validation.n_folds}_{metric}"] = cv[f"test_{metric}"].mean()
    except:
        print("CV failed")

    submit_df = make_submit(model, X_preprocessed, y, X_test_preprocessed, index=X_test.record_id.values)
    log_artifacts(meta, model, X_preprocessed, submit_df)


if __name__ == "__main__":
    cfg: MLConfig = load_config()
    train_model(cfg)
