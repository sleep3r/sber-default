import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import lightgbm

from config import MLConfig, object_from_dict


def validate(probas: np.ndarray, y_val: np.ndarray, cutoff: float) -> Dict[str, float]:
    metrics = {}
    preds = (probas[:, 1] > cutoff).astype(int)

    metrics["clf_report"] = classification_report(y_val, preds)
    metrics["accuracy"] = accuracy_score(y_val, preds)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_val, preds)
    metrics["roc_auc"] = roc_auc_score(y_val, preds)
    metrics["precision"] = precision_score(y_val, preds)
    metrics["recall"] = recall_score(y_val, preds)
    metrics["F1"] = f1_score(y_val, preds)
    return metrics


class BaseCV:
    def __init__(
            self, cfg: MLConfig,
            X, y, train_features,
            X_to_pred=None, y_for_split=None,
            groups_for_split=None, weights=None, weights2=None,
            out_metric=roc_auc_score, base_train_seed=42, num_train_seeds=1,
            model_params=None,
            k_fold_fn=None, fold_seed=15, nfolds=5,
            save_model_name='', model_folder='../models/', use_saved_model=False,
            mask=0, mask_name='', train_kick_mask=np.array([]), verbose=True,
            cat_features=None, fl_multiclass=False
    ):
        self.cfg = cfg
        self.X = X
        self.y = y
        self.train_features = train_features
        self.X_to_pred = pd.DataFrame() if X_to_pred is None else X_to_pred
        self.y_for_split = y_for_split
        self.groups_for_split = groups_for_split
        self.weights = weights
        self.weights2 = weights2
        self.out_metric = out_metric
        self.base_train_seed = base_train_seed
        self.num_train_seeds = num_train_seeds
        self.model_type = cfg.model.type.split(".")[0]
        self.model_params = {} if model_params is None else model_params
        self.k_fold_fn = StratifiedKFold if k_fold_fn is None else k_fold_fn
        self.fold_seed = fold_seed
        self.nfolds = nfolds
        self.save_model_name = save_model_name
        self.model_folder = model_folder
        self.use_saved_model = use_saved_model
        self.mask = mask
        self.mask_name = mask_name
        self.train_kick_mask = train_kick_mask
        self.verbose = verbose
        self.cat_features = [] if cat_features is None else cat_features
        self.fl_multiclass = fl_multiclass

    def run(self):
        start_time_cross_val = time.time()

        if self.save_model_name != '':
            try:
                os.mkdir(self.model_folder + self.save_model_name)
            except Exception:
                pass

        if self.weights is None:
            weights = np.ones(shape=(self.X.shape[0],))
        else:
            weights = np.array(self.weights)

        if self.weights2 is None:
            weights2 = weights
        else:
            weights2 = np.array(self.weights2)

        if self.y_for_split is None:
            y_for_split = self.y

        pred_cols = ['predictions']
        if self.fl_multiclass:
            nclasses = self.model_params['num_class']
            pred_cols = ['pred_cl' + str(i) for i in range(nclasses)]

        flag_save_model_features = True
        train_score_mask = ''
        cv_score_mask = 0
        kick = ''
        use_mask2 = False
        log_mask = ''
        fold_score_mask = ''
        if np.sum(self.mask) > 0:
            log_mask = 'mask:'
            use_mask2 = True
            mask = np.array(self.mask)

        cat_feats_ind = [i for i, j in enumerate(self.train_features) if j in self.cat_features]

        self.X["ID"] = self.X.reset_index(drop=True).index
        self.X_to_pred['ID'] = self.X_to_pred.reset_index(drop=True).index

        train_predictions = pd.DataFrame()
        train_predictions['ID'] = self.X['ID'].values
        test_predictions = pd.DataFrame([], columns=['ID', 'predictions'])
        if not self.X_to_pred.empty:
            test_predictions['ID'] = np.array(self.X_to_pred['ID'])

        for f in pred_cols:
            train_predictions[f] = 0
            if not self.X_to_pred.empty:
                test_predictions[f] = 0

        if self.fl_multiclass:
            y_predicted_train_avg = np.full(shape=(self.X.shape[0], nclasses), fill_value=0, dtype=float)
            y_predicted_test = np.zeros(shape=(self.X_to_pred.shape[0], nclasses))

        else:
            y_predicted_train_avg = np.full(shape=(self.X.shape[0],), fill_value=0, dtype=float)
            y_predicted_test = np.zeros(shape=(self.X_to_pred.shape[0],))

        if 'fold' in self.X.columns:
            X = self.X[self.train_features + ['fold']].apply(lambda x: np.round(x, 6) if 'float' in str(x.dtype) else x)
        else:
            X = self.X[self.train_features].apply(lambda x: np.round(x, 6) if 'float' in str(x.dtype) else x)

        if not self.X_to_pred.empty:
            X_to_pred = self.X_to_pred[self.train_features].apply(
                lambda x: np.round(x, 6) if 'float' in str(x.dtype) else x)

        if self.k_fold_fn == GroupKFold:
            k_fold = GroupKFold(n_splits=self.nfolds)
        else:
            k_fold = self.k_fold_fn(n_splits=self.nfolds, random_state=self.fold_seed, shuffle=True)

        # k_fold = TimeSeriesSplit(n_splits=nfolds+3)

        # k_fold =  k_fold_fn(n_splits=nfolds, random_state=fold_seed, shuffle=True)
        if (self.verbose):
            print(X[self.train_features].shape)

        for num in range(self.num_train_seeds):
            start_time_train_seed = time.time()
            train_seed = self.base_train_seed + 10 * num
            cv_score = []
            cv_score_mask = []

            i = 0
            if self.fl_multiclass:
                y_predicted_train = np.full(shape=(X.shape[0], nclasses), fill_value=-1, dtype=float)
            else:
                y_predicted_train = np.full(shape=(X.shape[0],), fill_value=-1, dtype=float)

            for train_index, test_index in k_fold.split(X, y_for_split, groups=self.groups_for_split):
                time_start_fold = time.time()

                i += 1

                model_full_name = self.model_folder + self.save_model_name + '/fold' + str(i) + '_seed' + str(
                    train_seed) + '.model'

                X_train, X_test = X[self.train_features].iloc[train_index], X[self.train_features].iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                weights_train, weights_test = weights2[train_index], weights[test_index]
                if use_mask2:
                    mask2 = mask[test_index]

                if len(self.train_kick_mask) > 0:
                    inv_train_kick_mask_fold = ~np.array(self.train_kick_mask.iloc[train_index])
                    X_train = X_train[inv_train_kick_mask_fold]
                    y_train = y_train[inv_train_kick_mask_fold]
                    weights_train = weights_train[inv_train_kick_mask_fold]
                    kick = 'kicked:' + str((~inv_train_kick_mask_fold).sum())

                if self.model_type == 'lightgbm':
                    if self.use_saved_model:
                        model = lightgbm.Booster(model_file=model_full_name)
                        best_iter = -1
                    else:
                        self.model_params['categorical_column'] = cat_feats_ind
                        model = lightgbm.LGBMModel(**self.model_params)
                        model.random_state = train_seed
                        e_stop = round(5 / model.get_params()['learning_rate'])
                        model.fit(X_train, np.array(y_train), sample_weight=weights_train,
                                  eval_set=(X_test, y_test),
                                  early_stopping_rounds=e_stop, eval_metric=model.metric,  # lgb_wrmse,
                                  verbose=False)
                        best_iter = model.best_iteration_

                        if self.save_model_name != '':
                            model.booster_.save_model(model_full_name)
                            if flag_save_model_features:
                                flag_save_model_features = False
                                pd.DataFrame(columns=self.train_features).to_csv(
                                    self.model_folder + self.save_model_name + '/train_features.txt', index=False)
                else:
                    model = object_from_dict(self.cfg.model)
                    fit_params = self.cfg.model.get("fit_params", {})
                    if self.cfg.model.eval_set_param:
                        fit_params[self.cfg.model.eval_set_param] = (X_test, y_test)
                    model.fit(X_train, np.array(y_train), **fit_params)
                    best_iter = -1

                #################### predict oof and test ########################
                try:
                    if not self.fl_multiclass:
                        y_pred_fold = model.predict_proba(X_test)[:, 1]
                    else:
                        y_pred_fold = model.predict_proba(X_test)

                except Exception:
                    if self.model_type == 'xgb':
                        y_pred_fold = model.predict(X_test, ntree_limit=best_iter)
                    elif self.model_type == 'cgb':
                        y_pred_fold = model.predict(X_test, prediction_type='Probability')
                    else:
                        y_pred_fold = model.predict(X_test)

                y_predicted_train[test_index,] = y_pred_fold.copy()
                fold_score = round(self.out_metric(round(y_test), y_pred_fold, sample_weight=weights_test), 5)

                if not X_to_pred.empty:
                    try:
                        if not self.fl_multiclass:
                            if self.model_type == 'xgb':
                                y_predicted_test += model.predict_proba(X_to_pred, ntree_limit=best_iter)[:, 1] / (
                                        self.nfolds * self.num_train_seeds)
                            elif self.model_type == 'cgb':
                                y_predicted_test += model.predict_proba(X_to_pred)[:, 1] / (
                                        self.nfolds * self.num_train_seeds)
                            else:
                                y_predicted_test += model.predict_proba(X_to_pred)[:, 1] / (
                                        self.nfolds * self.num_train_seeds)
                        else:
                            if self.model_type == 'xgb':
                                y_predicted_test += model.predict_proba(X_to_pred, ntree_limit=best_iter) / (
                                        self.nfolds * self.num_train_seeds)
                            elif self.model_type == 'cgb':
                                y_predicted_test += model.predict_proba(X_to_pred) / (
                                        self.nfolds * self.num_train_seeds)
                            else:
                                y_predicted_test += model.predict_proba(X_to_pred) / (
                                        self.nfolds * self.num_train_seeds)
                    except Exception:
                        if self.model_type == 'xgb':
                            y_predicted_test += model.predict(X_to_pred, ntree_limit=best_iter) / (
                                    self.nfolds * self.num_train_seeds)
                        elif self.model_type == 'cgb':
                            y_predicted_test += model.predict(X_to_pred, prediction_type='Probability') / (
                                    self.nfolds * self.num_train_seeds)
                        else:
                            y_predicted_test += model.predict(X_to_pred) / (self.nfolds * self.num_train_seeds)

                if use_mask2:
                    fold_score_mask = round(
                        self.out_metric(y_test[mask2], y_pred_fold[mask2], sample_weight=weights_test[mask2]), 5)
                    cv_score_mask.append(fold_score_mask)

                if (self.verbose):
                    print(i, "it:", best_iter, "score:", fold_score, log_mask, fold_score_mask,
                          'tot:', len(test_index), 'target:', round(np.sum(y_test)), kick,
                          'time {0}s'.format(round((time.time() - time_start_fold), 1)))
                cv_score.append(fold_score)

            y_predicted_train_avg += y_predicted_train / self.num_train_seeds
            train_score_avg = round(np.array(cv_score).mean(), 5)
            train_score = round(self.out_metric(round(self.y), y_predicted_train, sample_weight=weights), 5)
            train_score_std = round(np.array(cv_score).std(), 5)

            if use_mask2:
                train_score_mask = round(
                    self.out_metric(self.y[mask], y_predicted_train[mask], sample_weight=weights[mask]), 5)

            if (self.verbose):
                print('seed:', train_seed, 'metric_avg:', train_score_avg,
                      'metric_full:', train_score, 'folds_std:', train_score_std,
                      log_mask, train_score_mask,
                      'time: {0}s'.format(round((time.time() - start_time_train_seed), 1)))

        ################# save predictions for output #####################
        train_predictions.loc[:, pred_cols] = y_predicted_train_avg
        if not X_to_pred.empty:
            test_predictions.loc[:, pred_cols] = np.array(y_predicted_test)

        ############# calc stat for multi train seeds ####################
        if self.num_train_seeds > 1:
            cv_score = []
            cv_score_mask = []

            for train_index, test_index in k_fold.split(X, y_for_split, groups=self.groups_for_split):
                y_test = self.y.iloc[test_index]
                y_pred_fold = y_predicted_train_avg[test_index]
                fold_score = round(self.out_metric(round(y_test), y_pred_fold, sample_weight=weights[test_index]), 5)
                cv_score.append(fold_score)
                if use_mask2:
                    mask2 = mask[test_index]
                    fold_score_mask = round(
                        self.out_metric(y_test[mask2], y_pred_fold[mask2], sample_weight=weights[test_index][mask2]), 5)
                    cv_score_mask.append(fold_score_mask)

            train_score = round(self.out_metric(round(self.y), y_predicted_train_avg, sample_weight=weights), 5)
            train_score_avg = round(np.array(cv_score).mean(), 5)
            train_score_std = round(np.array(cv_score).std(), 5)

            if use_mask2:
                train_score_mask = round(
                    self.out_metric(self.y[mask], y_predicted_train_avg[mask], sample_weight=weights[mask]), 5)

            if (self.verbose):
                print('All:', 'metric_avg:', train_score_avg, 'metric_full:', train_score,
                      'folds_std:', train_score_std, log_mask, train_score_mask,
                      'time: {0}s'.format(round((time.time() - start_time_cross_val), 1)))

        return (np.array(cv_score), train_score, train_score_std,
                # np.array(cv_score_mask), train_score_mask,
                train_predictions, test_predictions)
