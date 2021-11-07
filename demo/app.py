import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="GSBxSber Hackathon Demo")

st.title('GSBxSber Hackathon Demo')
st.write('''Realtime scoring API default prediction model''')


def user_input_features():
    st.sidebar.header('Company features')

    input_features = {}
    input_features['model_type'] = st.sidebar.selectbox("Model type:", ['fin', 'no_fin'])

    if input_features['model_type'] == "fin":
        input_features["ul_capital_sum"] = st.sidebar.number_input('ul_capital_sum', 0, step=1)
        input_features["ul_founders_cnt"] = st.sidebar.number_input('ul_founders_cnt', 0, step=1)
    elif input_features['model_type'] == "no_fin":
        input_features["ul_capital_sum"] = st.sidebar.number_input('ul_capital_sum', 0, step=1)
    return [input_features]


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# explain model prediction results
def explain_model_prediction():
    # Calculate Shap values
    X, y = shap.datasets.diabetes()
    model = RandomForestClassifier()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    p = shap.force_plot(explainer.expected_value[1], shap_values[1], X)
    return p, shap_values, X


json_data = user_input_features()

submit = st.sidebar.button('Get predictions')
p, shap_values, X = explain_model_prediction()
st.subheader('Model Prediction Interpretation Plot')
st_shap(p, height=500)

st.subheader('Summary Plot 1')
fig, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[1], X)
st.pyplot(fig)

st.subheader('Summary Plot 2')
fig, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[1], X, plot_type='bar')
st.pyplot(fig)
