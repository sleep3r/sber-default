import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import load_config, MLConfig, object_from_dict
from dataset import DefaultDataset
from preprocessing import DefaultTransformer
from validation import get_train_folds, validate
from utils.env import collect_env
from utils.path import mkdir_or_exist


def create_workdir(cfg: MLConfig, meta: dict) -> dict:
    dirname = f"{cfg.exp_name}_{datetime.now().strftime('%d.%m/%H:%M:%S')}"
    meta["exp_dir"] = Path(cfg.work_dir) / dirname
    mkdir_or_exist(meta["exp_dir"])
    return meta


def env_collect(cfg: MLConfig, meta: dict) -> dict:
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'

    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
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


def log_report(meta: dict):
    print("\nClassification report best model:")
    print(meta["best_metrics"].pop("clf_report"))

    print("Metrics:")
    for name, value in meta["best_metrics"].items():
        print(f"{name} = {value}")


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

    print("\nSplitting dataset...")
    train_folds = get_train_folds(cfg, X_preprocessed, y)

    print("\nTraining...")
    model = object_from_dict(cfg.model)
    best_metric = 0
    best_estimator = None
    for i, train_fold in tqdm(enumerate(train_folds), total=len(train_folds)):
        X_train, y_train = train_fold[0]
        X_val, y_val = train_fold[1]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = validate(meta, preds=preds, y_val=y_val, fold_id=i)

        if metrics[cfg.validation.params.scoring] > best_metric:
            best_estimator = model
            best_metric = metrics[cfg.validation.params.scoring]
            meta["best_metrics"] = metrics

    log_report(meta)


if __name__ == "__main__":
    cfg: MLConfig = load_config()
    train_model(cfg)
