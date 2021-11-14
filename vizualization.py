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
                       base_features: list, file_name: str,file_name2: str, num_points_to_shap=1000 ) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[base_features].iloc[:num_points_to_shap, ])
    s_plot = plt.figure()
    plt.subplot(311)
    shap.summary_plot(shap_values, X_train[base_features].iloc[:num_points_to_shap, ], show=False)

    plt.subplot(312)
    shap.summary_plot(shap_values[1], X_train[base_features].iloc[:num_points_to_shap, ],
                             plot_type="layered_violin", show=False)
    plt.subplot(313)
    shap.summary_plot(shap_values[1], X_train[base_features].iloc[:num_points_to_shap, ],
                             max_display=len(base_features), auto_size_plot=True, show=False)
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

    plt.savefig(file_name2)

def plots_for_distr(X: pd.DataFrame, y: np.array, file_name: str):
    def facetting_scatter_plot(df,y, columns, n_cols=3):
        numeric_cols = columns
        n_rows = -(-len(numeric_cols) // n_cols)  # math.ceil in a fast way, without import
        row_pos, col_pos = 1, 0
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)
        for col in numeric_cols:
            # trace extracted from the fig
            trace = px.histogram(df, x=col, color=y, marginal="violin")["data"]
            # auto selecting a position of the grid
            if col_pos == n_cols:
                row_pos += 1
            col_pos = col_pos + 1 if (col_pos < n_cols) else 1

            for traces in trace:
                fig.append_trace(traces, row=row_pos, col=col_pos)
        #    fig.add_trace(trace[0], row=row_pos, col=col_pos)
        #    fig.add_trace(trace[1], row=row_pos, col=col_pos)
        return fig
    # fig, axes = plt.subplots(nrows=7, ncols=7, figsize=(20, 14))
    # axes = axes.ravel()
    #this_figure = make_subplots(rows=5, cols=5,shared_xaxes=False)

    num_features = ['bus_age','ogrn_age','adr_actual_age','head_actual_age','cap_actual_age']
    # for i in range(5):
    #     for j in range(5):
    #         col = X.columns[i*5+j]
    #         if col not in num_features:
    #             continue
    #         fig = px.histogram(X, x=col, color=y, marginal="violin")#,
    #                                     #hover_data=X.columns,  width=500, height=500)
    #         figure1_traces = []
    #
    #         for trace in range(len(fig["data"])):
    #             figure1_traces.append(fig["data"][trace])
    #         for traces in figure1_traces:
    #             this_figure.append_trace(traces, row=i+1, col=j+1)

    #final_graph = dcc.Graph(figure=this_figure)


    fig = facetting_scatter_plot(df = X,y=y, columns=num_features,n_cols=3)
    fig.update_layout(width=1000, height=800, title='% of variation in Happiness Score explained by each feature',
                      title_x=0.5)
    plotly.offline.plot(fig, filename='c:/Users/User/ipynbs/lifeExp.png')

#fig = make_subplots(rows=2, cols=1, shared_xaxes=False)


