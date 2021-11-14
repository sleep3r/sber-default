import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import MLConfig, object_from_dict
import shap
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go


def shap_vizualization(model, X_train: pd.DataFrame, y_train : np.array, X_test: pd.DataFrame, y_test: np.array,
                       base_features: list, file_name: str,file_name2: str, num_points_to_shap=1000,model_type='forest' ) -> None:
    s_plot = plt.figure(figsize=(20, 14))
    plt.subplot(411)
    if np.isin(model_type, ['forest','lgbm_fin','lgbm_nofin','forest_nofin']):
        df = pd.DataFrame({'cols' : X_train.columns, 'feature_value': model.feature_importances_}).\
            sort_values(by='feature_value')
        plt.barh(df.cols, df.feature_value)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[base_features].iloc[:num_points_to_shap, ])

    plt.subplot(412)
    shap.summary_plot(shap_values, X_train[base_features].iloc[:num_points_to_shap, ], show=False)

    plt.subplot(413)
    shap.summary_plot(shap_values[1], X_train[base_features].iloc[:num_points_to_shap, ],
                             plot_type="layered_violin", show=False, plot_size=[8,6])
    plt.subplot(414)
    shap.summary_plot(shap_values[1], X_train[base_features].iloc[:num_points_to_shap, ],
                             max_display=len(base_features),  show=False,plot_size=[8,6])
    plt.tight_layout()
    s_plot.savefig(file_name)
    del s_plot

    shap_sum = np.abs(shap_values[1]).mean(axis=0)
    importance_df = pd.DataFrame([base_features, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
    importance_cols = importance_df['column_name'].loc[0:15]

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 14))
    axes = axes.ravel()

    for i, col in enumerate(importance_cols):
        shap.dependence_plot(col, shap_values[1], X_train[base_features].iloc[:num_points_to_shap, ],
                             ax=axes[i], show=False)
    plt.tight_layout()
    plt.savefig(file_name2, dpi=200)


def plots_for_distr(X: pd.DataFrame, y: np.array, file_name: str):
    def facetting_scatter_plot(df,y, columns, n_cols=3):
        numeric_cols = columns
        n_rows = -(-len(numeric_cols) // n_cols)
        row_pos, col_pos = 1, 0
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)
        for col in numeric_cols:
            trace = px.histogram(df, x=col, color=y, marginal="violin",hover_data=df.columns)["data"]
            if col_pos == n_cols:
                row_pos += 1
            col_pos = col_pos + 1 if (col_pos < n_cols) else 1

            for traces in trace:
                fig.append_trace(traces, row=row_pos, col=col_pos)
        return fig
    num_features = ['bus_age','ogrn_age','adr_actual_age','head_actual_age','cap_actual_age']
    fig = facetting_scatter_plot(df = X,y=y, columns=num_features,n_cols=3)
    fig.update_layout(width=1000, height=800,
                      title_x=0.5)
    plotly.offline.plot(fig, filename=file_name)



