import gc
from functools import reduce
from pathlib import Path
from typing import List, Type

import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st
import shap
import matplotlib.pyplot as plt

from demo.model import LGBMCVModel

st.set_page_config(page_title="GSBxSber Hackathon Demo")
st.title('GSBxSber Hackathon Demo')
st.write('''Realtime scoring API default prediction model''')


def input(cv_model: LGBMCVModel, model_type: str) -> (List[dict], bool):
    """
    Sets features input fields based on model

    Args:
        cv_model (LGBMCVModel): cross-validated model;
        model_type (str): model type.
    Returns:
        input_features (List[dict]): list of input fields;
        submit (bool): submit button state.
    """
    input_features = {}
    form = st.sidebar.form(key=model_type)
    for feature_name in [*cv_model.models.values()][0].feature_name():
        input_features[feature_name] = form.number_input(feature_name, value=np.nan)
    submit = form.form_submit_button('Get predictions')
    return [input_features], submit


def plot_shap_graphs(shap_values: np.ndarray, X_test: pd.DataFrame) -> None:
    """
    Plots all SHAP graphs.

    Args:
        shap_values (np.ndarray): SHAP values;
        X_test (pd.DataFrame): test set.
    """
    st.subheader('SHAP Summary Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test)
    st.pyplot(fig)

    st.subheader('SHAP Summary Bar Plot')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test, plot_type='bar')
    st.pyplot(fig)


def plot_prediction(
        submit: bool,
        explainer: Type[shap.Explainer],
        X_test: pd.DataFrame,
        shap_values: np.ndarray
) -> None:
    """
    Plots SHAP interpretation plot.

    Args:
        submit (bool): submit button state;
        explainer (Type[shap.Explainer]): explainer object;
        X_test (pd.DataFrame): test set;
        shap_values (np.ndarray): SHAP values.
    """
    if submit:
        st.subheader('Model Prediction Interpretation Plot')
        shap.force_plot(
            explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :],
            matplotlib=True, show=False, figsize=(28, 4)
        )
        st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)


def plot_importance(cv_model: LGBMCVModel) -> None:
    """
    Plots SHAP feature importance plots.

    Args:
        cv_model (LGBMCVModel): cross-validated model.
    """
    st.subheader('Feature Importance')
    importance_type = st.selectbox('Select the desired importance type', ('auto', 'split', 'gain'), index=0)
    ax = lgb.plot_importance(
        [*cv_model.models.values()][0],
        importance_type=importance_type,
    )
    fig = plt.gcf()
    st.pyplot(fig)


def plot_model_tree(cv_model: LGBMCVModel) -> None:
    """
    Plots model boosting tree.

    Args:
        cv_model (LGBMCVModel): cross-validated model.
    """
    st.subheader('Model Tree')
    ax = lgb.plot_tree(
        [*cv_model.models.values()][0],
        tree_index=1,
        show_info=['split_gain'],
        dpi=600
    )
    fig = plt.gcf()
    st.pyplot(fig, dpi=600)


def plot_dependence(shap_values: np.ndarray, X_test: pd.DataFrame) -> None:
    """
    Plots SHAP dependence plots for common features.

    Args:
        shap_values (np.ndarray): SHAP values;
        X_test (pd.DataFrame): test set.
    """
    for feature_name in X_test.columns:
        shap.dependence_plot(feature_name, shap_values[1], X_test, show=False)
        st.pyplot()
        plt.clf()


@st.cache(suppress_st_warning=True, show_spinner=False)
def explain_model(checkpoint_path: str) -> (np.ndarray, pd.DataFrame, Type[shap.Explainer], LGBMCVModel):
    """
    Base func for model init and SHAP explanation run.

    Args:
        checkpoint_path (str): path to model checkpoint files.
    """
    cv_model = LGBMCVModel(checkpoint_path)
    all_shap_values = []
    test_sets = []

    for model_name, fold_model in cv_model.models.items():
        explainer = shap.TreeExplainer(fold_model)

        X_test = pd.read_csv(Path(checkpoint_path) / f"{model_name}.csv", index_col=0)
        repl = {
            np.inf: 0,
            -np.inf: 0
        }
        X_test = X_test.replace(repl).copy()
        shap_values = explainer.shap_values(X_test)

        all_shap_values.append(shap_values)
        test_sets.append(X_test)

    X_test = reduce(lambda a, b: a.append(b), test_sets)
    shap_values = np.concatenate(all_shap_values, axis=1)
    return shap_values, X_test, explainer, cv_model


model_type = st.sidebar.selectbox("Model:", ['fin', 'no_fin'])
st.sidebar.header('Company features')
if model_type == "fin":
    shap_values, X_test, explainer, cv_model = explain_model(
        "/app/checkpoints/fin"
    )
    json_data, submit = input(cv_model, model_type)
    plot_prediction(submit, explainer, X_test, shap_values)
    plot_importance(cv_model)
    plot_shap_graphs(shap_values, X_test)
    plot_model_tree(cv_model)
    plot_dependence(shap_values, X_test)
else:
    shap_values, X_test, explainer, cv_model = explain_model(
        "/app/checkpoints/no_fin"
    )
    json_data, submit = input(cv_model, model_type)
    plot_prediction(submit, explainer, X_test, shap_values)
    plot_importance(cv_model)
    plot_shap_graphs(shap_values, X_test)
    plot_dependence(shap_values, X_test)
    plot_dependence(shap_values, X_test)
