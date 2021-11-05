import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split

from config import load_config, MLConfig, object_from_dict
from dataset import DefaultDataset
from preprocessing import DefaultTransformer
from features import DefaultGenerator, DefaultSelector
from validation import get_train_folds, validate
from utils.env import collect_env
from utils.path import mkdir_or_exist

warnings.simplefilter('ignore')


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


def make_submit(model, X: pd.DataFrame, y, X_test, index: np.ndarray, cutoff: float) -> pd.DataFrame:
    model.fit(X.values, y)

    predict = model.predict_proba(X_test.values)
    answ_df = pd.DataFrame(index, columns=["id"])
    answ_df['predict'] = (predict[:, 1] > cutoff).astype(int)
    return answ_df


def log_dataset(X: pd.DataFrame, y, X_test, exp_dir: Path) -> None:
    X["target"] = y.values
    X.to_csv(exp_dir / "data_train.csv", index=None)
    X_test.to_csv(exp_dir / "data_test.csv", index=None)


def log_submit(submit_df: pd.DataFrame, exp_dir: Path) -> None:
    submit_df.to_csv(exp_dir / 'PD-submit.csv', index=False, sep=';')


def log_artifacts(meta: dict, best_estimator, X: pd.DataFrame, X_test, y, submit_df: pd.DataFrame) -> None:
    exp_dir = meta.pop("exp_dir")

    log_report(meta, exp_dir)
    log_best_model(best_estimator, exp_dir)
    log_dataset(X, y, X_test, exp_dir)
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

    print("\nGenerating features...")
    generator = DefaultGenerator(cfg, X, X_test)
    X_generated, X_test_generated = generator.generate_features()

    print("\nPreprocessing dataset...")
    transformer = DefaultTransformer(cfg, X_generated, X_test_generated)
    preprocessed = transformer.transform()
    (X_generated_preprocessed, y), X_test_generated_preprocessed = preprocessed["train"], preprocessed["test"]

    print("\nSelecting features...")
    selector = DefaultSelector(cfg, X_generated_preprocessed, X_test_generated_preprocessed)
    X_generated_preprocessed_selected, X_test_generated_preprocessed_selected = selector.select_features()

    print("\nTraining...")
    model = object_from_dict(cfg.model)

    X_train, X_val, y_train, y_val = train_test_split(
        X_generated_preprocessed_selected, y,
        stratify=y, test_size=cfg.validation.test_size, shuffle=True
    )

    print("val...")
    fit_params = cfg.model.get("fit_params", {})
    if cfg.model.eval_set_param:
        fit_params[cfg.model.eval_set_param] = [X_val, y_val]
    model.fit(X_train.values, y_train, **fit_params)
    probas = model.predict_proba(X_val.values)
    metrics = validate(probas=probas, y_val=y_val, cutoff=cfg.validation.cutoff)
    meta["metrics"] = metrics

    try:
        print("crossval...")
        cv = cross_validate(
            object_from_dict(cfg.model), X_generated_preprocessed_selected.values, y,
            cv=cfg.validation.n_folds, scoring=cfg.validation.scoring,
            fit_params=fit_params
        )
        for metric in cfg.validation.scoring:
            meta["metrics"][f"CV_{cfg.validation.n_folds}_{metric}"] = cv[f"test_{metric}"].mean()
    except:
        print("CV failed")

    submit_df = make_submit(
        model, X_generated_preprocessed_selected, y, X_test_generated_preprocessed_selected,
        index=X_test.record_id.values, cutoff=cfg.validation.cutoff
    )
    log_artifacts(
        meta, model, X_generated_preprocessed_selected, X_test_generated_preprocessed_selected, y, submit_df
    )


if __name__ == "__main__":
    cfg: MLConfig = load_config()
    train_model(cfg)
