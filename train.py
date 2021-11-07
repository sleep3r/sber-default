import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config, MLConfig, object_from_dict
from dataset import DefaultDataset
from preprocessing import DefaultTransformer
from features import DefaultGenerator, DefaultSelector
from validation import validate, BaseCV
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


def make_submit(cfg, test_predictions: pd.DataFrame, X_test, cutoff: float) -> pd.DataFrame:
    probas = test_predictions.predictions.values

    if cfg.preprocessing.select_rows is not None:
        index = X_test[X_test[cfg.preprocessing.select_rows]].record_id.values
    else:
        index = X_test.record_id.values

    answ_df = pd.DataFrame(index, columns=["id"])
    answ_df['predict'] = (probas > cutoff).astype(int)
    return answ_df


def log_dataset(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, exp_dir: Path) -> None:
    X["target"] = y.values
    X.to_csv(exp_dir / "data_train.csv", index=None)  # noqa
    X_test.to_csv(exp_dir / "data_test.csv", index=None)  # noqa


def log_submit(submit_df: pd.DataFrame, exp_dir: Path) -> None:
    submit_df.to_csv(exp_dir / 'PD-submit.csv', index=False, sep=';')  # noqa


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

    print("val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_generated_preprocessed_selected, y,
        stratify=y, test_size=cfg.validation.test_size, shuffle=True
    )

    fit_params = cfg.model.get("fit_params", {})
    if cfg.model.eval_set_param:
        fit_params[cfg.model.eval_set_param] = (X_val, y_val)

    model.fit(X_train.values, y_train, **fit_params)
    probas = model.predict_proba(X_val.values)
    metrics = validate(probas=probas, y_val=y_val, cutoff=cfg.validation.cutoff)
    meta["metrics"] = metrics

    print("crossval...")
    cv_params = cfg.validation.cv_params
    cv = BaseCV(
        cfg, X_generated_preprocessed_selected, y,
        train_features=X_generated_preprocessed_selected.columns,
        X_to_pred=X_test_generated_preprocessed_selected, out_metric=object_from_dict(cv_params.out_metric),
        base_train_seed=cfg.seed, fold_seed=cv_params.fold_seed,
        num_train_seeds=cv_params.num_train_seeds,
        model_params=cfg.model.params,
        nfolds=cv_params.n_folds,
        cat_features=cfg.preprocessing.cat_features,
        verbose=cv_params.verbose,
        k_fold_fn=object_from_dict(cv_params.k_fold_fn),
        groups_for_split=X_generated_preprocessed[cv_params.groups_col] if cv_params.groups_col else None,
        save_model_name=cfg.exp_name,
        model_folder=meta["exp_dir"]
    )

    cv_score, train_score, train_score_std, train_predictions, test_predictions = cv.run()
    meta["metrics"]["CV_score"] = cv_score.mean()
    meta["metrics"]["train_score"] = train_score
    meta["metrics"]["train_score_std"] = train_score_std

    submit_df = make_submit(
        cfg=cfg,
        test_predictions=test_predictions, X_test=X_test_generated,
        cutoff=cfg.validation.cutoff
    )
    log_artifacts(
        meta=meta, best_estimator=model,
        X=X_generated_preprocessed_selected, X_test=X_test_generated_preprocessed_selected, y=y,
        submit_df=submit_df
    )


if __name__ == "__main__":
    cfg: MLConfig = load_config()
    train_model(cfg)
