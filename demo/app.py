from functools import reduce
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from model import LGBMCVModel

st.set_page_config(page_title="GSBxSber Hackathon Demo")
st.title('GSBxSber Hackathon Demo')
st.write('''Realtime scoring API default prediction model''')


def input(cv_model: LGBMCVModel, model_type: str):
    input_features = {}
    form = st.sidebar.form(key=model_type)
    for feature_name in [*cv_model.models.values()][0].feature_name():
        input_features[feature_name] = form.number_input(feature_name)
    submit = form.form_submit_button('Get predictions')
    return [input_features], submit


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def plot_shap_graphs(shap_values, X_test):
    st.subheader('Summary Plot 1')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test)
    st.pyplot(fig)

    st.subheader('Summary Plot 2')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test, plot_type='bar')
    st.pyplot(fig)


def plot_prediction(submit, explainer, X_test, shap_values):
    if submit:
        st.subheader('Model Prediction Interpretation Plot')
        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :], matplotlib=True)
        st.pyplot(fig)


def plot_model(cv_model):
    st.subheader('Model tree')
    ax = lgb.plot_tree(
        [*cv_model.models.values()][0],
        tree_index=1,
        show_info=['split_gain'],
        figsize=(20, 15)
    )
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    st.pyplot(fig)


@st.cache(suppress_st_warning=True, show_spinner=False)
def explain_model(checkpoint_path: str):
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
    plot_shap_graphs(shap_values, X_test)
    plot_model(cv_model)

else:
    shap_values, X_test, explainer, cv_model = explain_model(
        "/app/checkpoints/no_fin"
    )
    json_data, submit = input(cv_model, model_type)
    plot_prediction(submit, explainer, X_test, shap_values)
    plot_shap_graphs(shap_values, X_test)
    plot_model(cv_model)
