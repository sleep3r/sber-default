from functools import reduce
from pathlib import Path

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


def fin_input():
    input_features = {}
    form = st.sidebar.form(key="fin_form")
    input_features["ul_capital_sum"] = form.number_input('ul_capital_sum', 0, step=1)
    input_features["ul_founders_cnt"] = form.number_input('ul_founders_cnt', 0, step=1)
    submit = form.form_submit_button('Get predictions')
    return [input_features], submit


def no_fin_input():
    input_features = {}
    form = st.sidebar.form(key="no_fin_form")
    input_features["ul_capital_sum"] = form.number_input('ul_capital_sum', 0, step=1)
    submit = form.form_submit_button('Get predictions')
    return [input_features], submit


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def plot_graphs(shap_values, X_test, explainer, submit):
    if submit:
        st.subheader('Model Prediction Interpretation Plot')
        p = shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :])
        st_shap(p, height=150)

    st.subheader('Summary Plot 1')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test)
    st.pyplot(fig)

    st.subheader('Summary Plot 2')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values[1], X_test, plot_type='bar')
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
    return shap_values, X_test, explainer


model_type = st.sidebar.selectbox("Model:", ['fin', 'no_fin'])
st.sidebar.header('Company features')
if model_type == "fin":
    json_data, submit = fin_input()
    shap_values, X_test, explainer = explain_model("~/checkpoints/fin")
    plot_graphs(shap_values, X_test, explainer, submit)
else:
    json_data, submit = no_fin_input()
    shap_values, X_test, explainer = explain_model("~/checkpoints/no_fin")
    plot_graphs(shap_values, X_test, explainer, submit)
