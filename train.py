import os
import random

import numpy as np
from tqdm import tqdm

from config import load_config, MLConfig, object_from_dict
from dataset import DefaultDataset
from preprocessing import DefaultTransformer
from validation import get_train_folds
from utils.env import collect_env
from utils.path import mkdir_or_exist


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
        print(f'\nSet random seed to {cfg.seed}\n')
        set_random_seed(cfg.seed)

    meta['seed'] = cfg.seed
    meta['exp_name'] = cfg.exp_name
    return meta


def prepare_training(cfg: MLConfig) -> dict:
    meta = dict()

    # create work_dir
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    # dump config
    cfg.dump(os.path.join(cfg.work_dir, "config.yml"))

    # log env info
    env_collect(cfg, meta)

    # log some basic info
    cfg.pretty_print()

    # set random seeds
    determine_exp(cfg, meta)
    return meta


def train_model(cfg: MLConfig):
    meta: dict = prepare_training(cfg)

    print("Loading dataset...")
    dataset = DefaultDataset(cfg)
    X, X_test = dataset.data["train"], dataset.data["test"]

    print("Preprocessing dataset...")
    transformer = DefaultTransformer(cfg, X, X_test)
    preprocessed = transformer.transform()
    (X_preprocessed, y), X_test_preprocessed = preprocessed["train"], preprocessed["test"]

    print("Generating features...")
    # feature selection
    # pass

    print("Splitting dataset...")
    train_folds = get_train_folds(cfg, X_preprocessed, y)

    print("Training...")
    model = object_from_dict(cfg.model)
    for i, train_fold in tqdm(enumerate(train_folds), total=len(train_folds)):
        X_train, y_train = train_fold[0]
        X_val, y_val = train_fold[1]

        model.fit(X_train, y_train)


if __name__ == "__main__":
    cfg: MLConfig = load_config()
    train_model(cfg)
