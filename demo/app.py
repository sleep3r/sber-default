import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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


@st.cache(suppress_st_warning=True, show_spinner=False)
def explain_model():
    # Calculate Shap values
    X, y = shap.datasets.diabetes()
    model = RandomForestClassifier()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values, X


explainer, shap_values, X = explain_model()

st.sidebar.header('Company features')
model_type = st.sidebar.selectbox("Model:", ['fin', 'no_fin'])

if model_type == "fin":
    json_data, submit = fin_input()
else:
    json_data, submit = no_fin_input()

if submit:
    st.subheader('Model Prediction Interpretation Plot')
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], X)
    st_shap(p, height=500)

st.subheader('Summary Plot 1')
fig, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[1], X)
st.pyplot(fig)

st.subheader('Summary Plot 2')
fig, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[1], X, plot_type='bar')
st.pyplot(fig)
