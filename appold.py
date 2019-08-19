# -*- coding: utf-8 -*-
print("Preparing to start dash app ...")

print('Loading libraries...')
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd

from pathlib import Path
import pickle


# from rf_explain import *

from explainer_methods import *
from explainer_plots import *

import plotly.io as pio
pio.templates.default = "none"

print("loading DataExplainer object...")

TestDataExplainer = pickle.load(open(Path.cwd()/'titanic_explainer.pkl', 'rb'))

print('Loading Dash...')
app = dash.Dash(__name__)

app.config['suppress_callback_exceptions']=True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.title = 'Model Explainer'

server = app.server

NO_OF_INTERACTION_GRAPHS = 3
NO_OF_DEPENDENCE_GRAPHS = 3

print('Defining layout...')

model_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Model overview:'),
            html.Div([
                html.Label('Bin size:'),
                dcc.Slider(id='precision-binsize',
                            min = 0.01, max = 0.5, step=0.01, value=0.05,
                            marks={
                                    0.01: '0.01',
                                    0.05: '0.05',
                                    0.10: '0.10',
                                    0.33: '0.33',
                                    0.5: '0.5',
                                },
                            included=False,
                            ),
            ], style={'margin': 20}),
            dcc.Graph(id='precision-graph'),
            html.Div([
                html.Label('Cutoff:'),
                dcc.Slider(id='precision-cutoff',
                            min = 0.01, max = 0.99, step=0.01, value=0.5,
                            marks={
                                    0.01: '0.01',
                                    0.25: '0.25',
                                    0.50: '0.50',
                                    0.75: '0.75',
                                    0.99: '0.99',
                                },
                            included=False,
                            ),
            ], style={'margin': 20}),

        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(id="loading-confusionmatrix-graph",
                         children=[dcc.Graph(id='confusionmatrix-graph')]),
            dcc.RadioItems(
                id='confusionmatrix-normalize',
                options=[{'label': o, 'value': o}
                            for o in ['Counts', 'Normalized']],
                value='Counts',
                labelStyle={'display': 'inline-block'}
            ),
        ]),
        dbc.Col([
            dcc.Loading(id="loading-roc-auc-graph",
                         children=[dcc.Graph(id='roc-auc-graph')]),
        ]),
        dbc.Col([
            dcc.Loading(id="loading-pr-auc-graph",
                         children=[dcc.Graph(id='pr-auc-graph')]),
        ]),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Model importances:'),

            dcc.RadioItems(
                id='permutation-or-shap',
                options=[
                    {'label': 'Permutation Importances',
                     'value': 'permutation'},
                    {'label': 'SHAP values',
                     'value': 'shap'}
                ],
                value='shap',
                labelStyle={'display': 'inline-block'}
            ),
            daq.ToggleSwitch(
                id='group-categoricals',
                label='Group Categoricals',
            ),
            html.Div('Select max number of importances to display:'),

            dcc.Slider(id='importance-tablesize',
                        min = 1, max = len(TestDataExplainer.columns),
                        value=15),
            # dash_table.DataTable(
            #     id='importances_table',
            #     columns=[{'id': c, 'name': c}
            #                 for c in ['Feature', 'Importance']],
            # ),
            dcc.Graph(id='importances-graph'),

        ])
    ]),
], fluid=True)

contributions_tab = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Label('Fill specific index'),
            dbc.Input(id='input-index', placeholder="Fill in index here...",
                        debounce=True),
            html.Div([
                dcc.RangeSlider(
                    id='prediction-range-slider',
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=[0.5, 1.0],
                    allowCross=False,
                    marks={
                            0.0:'0.0',
                            0.1:'0.1',
                            0.2:'0.2',
                            0.3:'0.3',
                            0.4:'0.4',
                            0.5:'0.5',
                            0.6:'0.6',
                            0.7:'0.7',
                            0.8:'0.8',
                            0.9:'0.9',
                            1.0:'1.0',
                    }
                )
            ], style={'margin': 20}),
            html.Div([
                dcc.RadioItems(
                    id='include-labels',
                    options=[
                        {'label': 'Only Pos', 'value': 'pos'},
                        {'label': 'Only Neg', 'value': 'neg'},
                        {'label': 'Both/either', 'value': 'any'},
                    ],
                    value='any',
                    labelStyle={'display': 'inline-block'}
                )
            ], style={'margin': 20}),
            html.Div([
                html.Button('random index', id='index-button'),
            ], style={'margin': 30})
        ]),
        dbc.Col([
             dcc.Loading(id="loading-model-prediction",
                         children=[dcc.Markdown(id='model-prediction')]),
        ]),
        dcc.Store(id='index-store'),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label('Number of features to display:'),
                dcc.Slider(id='contributions-size',
                    min = 1, max = len(TestDataExplainer.columns),
                    marks={int(i) : str(int(i))
                                for i in np.linspace(
                                        1, len(TestDataExplainer.columns), 6)},
                    step = 1, value=15),
            ]),
            html.Div(id='contributions-size-display', style={'margin-top': 20})
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Loading(id="loading-contributions-graph",
                         children=[dcc.Graph(id='contributions-graph')])
            ], style={'margin': 30})

        ], width=6),
        dbc.Col([
            html.Div([
                html.Label("Plot \'what if?\' for columns:"),
                dcc.Dropdown(id='pdp-col',
                    options=[{'label': col, 'value':col}
                                for col in TestDataExplainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=TestDataExplainer.mean_abs_shap_df().Feature[0]),
                dcc.Loading(id="loading-pdp-graph",
                        children=[dcc.Graph(id='pdp-graph')]),
            ], style={'margin': 30})
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='contributions_table',
                columns=[{'id': c, 'name': c}
                            for c in ['Reason', 'Effect']],
            ),
        ])
    ]),
], fluid=True)



dependence_tab = dbc.Container([
     dbc.Row([
        dbc.Col([
            dcc.Graph(id='dependence-shap-scatter-graph',
                      figure=plotly_shap_scatter_plot(
                                TestDataExplainer.shap_values,
                                TestDataExplainer.X,
                                TestDataExplainer.importances_df(
                                    type='shap', topx=20)\
                                        ['Feature'].values.tolist()))
        ]),
        dbc.Col([
            dbc.Input(id='dependence-highlight-index',
                        placeholder="Highlight index...",
                        debounce=True),
            dcc.Dropdown(id='dependence-col',
                    options=[{'label': col, 'value':col}
                                for col in TestDataExplainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=TestDataExplainer.mean_abs_shap_df().Feature[0]),
            dcc.Dropdown(id='dependence-color-col',
                    options=[{'label': col, 'value':col}
                                for col in TestDataExplainer.columns]),
            dcc.Graph(id='dependence-graph')
        ])
    ]),
],  fluid=True)

interactions_tab = dbc.Container([
     dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='interaction-col',
                    options=[{'label': col, 'value':col}
                                for col in TestDataExplainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=TestDataExplainer.mean_abs_shap_df().Feature[0]),
            dcc.Graph(id='interaction-shap-scatter-graph')
        ]),
        dbc.Col([
            dbc.Input(id='interaction-highlight-index',
                        placeholder="Highlight index...",
                        debounce=True),
            dcc.Dropdown(id='interaction-interact-col',
                    options=[{'label': col, 'value':col}
                                for col in TestDataExplainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=TestDataExplainer.mean_abs_shap_df().Feature[0]),
            dcc.Graph(id='interaction-graph')
        ])
    ]),
],  fluid=True)

trees_tab = dbc.Container([
     dbc.Row([
        dbc.Col([
            dcc.Graph(id='tree-predictions-graph'),
            dash_table.DataTable(
                id='tree-predictions-table',
            ),
        ])
    ]),
],  fluid=True)


app.layout = dbc.Container([
    dbc.Row([html.H1('WW Sollicitatie Model')]),
    dcc.Tabs(
        [
            dcc.Tab(children=model_tab,
                    label='Model Overview',
                    id='model_tab'),
            dcc.Tab(children=dependence_tab,
                    label='Dependence Plots',
                    id='dependence_tab'),
            dcc.Tab(children=interactions_tab,
                    label='Interactions graphs',
                    id='interactions_tab'),
            dcc.Tab(children=contributions_tab,
                    label='Individual Contributions',
                    id='contributions_tab'),
            dcc.Tab(children=trees_tab,
                    label='Individual Trees',
                    id='trees_tab'),
        ],
        id="tabs",
        value='model_tab'
    )
],  fluid=True)


@app.callback(
    Output('precision-graph', 'figure'),
    [Input('precision-binsize', 'value'),
     Input('precision-cutoff', 'value')],
)
def update_precision_graph(bin_size, cutoff):
    plot = plotly_precision_plot(
                TestDataExplainer.precision_df(
                    bin_size=bin_size), cutoff=cutoff)
    return plot


@app.callback(
    [Output('confusionmatrix-graph', 'figure'),
     Output('roc-auc-graph', 'figure'),
     Output('pr-auc-graph', 'figure')],
    [Input('precision-cutoff', 'value'),
     Input('confusionmatrix-normalize', 'value')],
)
def update_precision_graph(cutoff, normalized):
    confmat_plot = plotly_confusion_matrix(
                TestDataExplainer.y,
                TestDataExplainer.pred_probas,
                cutoff=cutoff,
                normalized=normalized=='Normalized')
    roc_auc_plot = plotly_roc_auc_curve(TestDataExplainer.y,
                TestDataExplainer.pred_probas,
                cutoff=cutoff)
    pr_auc_plot = plotly_pr_auc_curve(TestDataExplainer.y,
                TestDataExplainer.pred_probas,
                cutoff=cutoff)
    return (confmat_plot, roc_auc_plot, pr_auc_plot)


@app.callback(
     Output('importances-graph', 'figure'),
    [Input('importance-tablesize', 'value'),
     Input('group-categoricals', 'value'),
     Input('permutation-or-shap', 'value')]
)
def update_importances(tablesize, cats, permutation_shap):
    df = TestDataExplainer.importances_df(
                type=permutation_shap, topx=tablesize, cats=cats)
    plot = plotly_importances_plot(df)
    return plot


@app.callback(
    Output('input-index', 'value'),
    [Input('index-button', 'n_clicks')],
    [State('prediction-range-slider', 'value'),
     State('include-labels', 'value')]
)
def update_input_index(n_clicks, slider_range, include):
    print('slider range:', slider_range[0], slider_range[1])
    y = None
    if include=='neg': y = 0
    elif include=='pos': y = 1
    idx = TestDataExplainer.random_index(y=y,
                                   pred_proba_min=slider_range[0],
                                   pred_proba_max=slider_range[1])
    if idx is not None:
        return idx
    raise PreventUpdate


@app.callback(
    Output('index-store', 'data'),
    [Input('input-index', 'value')],
    [State('index-store', 'data')]
)
def update_bsn_div(input_index, old_index):
    if str(input_index).isdigit() and int(input_index) <= len(TestDataExplainer):
        return int(input_index)
    else:
        return old_index


@app.callback(
    Output('contributions-size-display', 'children'),
    [Input('contributions-size', 'value')])
def display_value(contributions_size):
    return f"Displaying top {contributions_size} features."


@app.callback(
    [Output('model-prediction', 'children'),
     Output('contributions-graph', 'figure'),
     Output('contributions_table', 'data')],
    [Input('index-store', 'data'),
     Input('contributions-size', 'value')]
)
def update_output_div(idx, topx):
    X, y, pred_proba, shap_values = TestDataExplainer[idx]

    model_prediction = f"## Index: {idx}\n"\
                        + f"## Prediction: {np.round(100*pred_proba,2)}%\n",
    contrib_df = TestDataExplainer.contrib_df(idx, topx=topx)
    plot = plotly_contribution_plot(contrib_df)
    summary_table = get_contrib_summary_df(contrib_df).to_dict('records')

    return (model_prediction, plot, summary_table)


@app.callback(
    Output('pdp-graph', 'figure'),
    [Input('index-store', 'data'),
     Input('pdp-col', 'value')]
)
def update_pdp_graph(idx, col):
    pdp_result = TestDataExplainer.pdp(col)

    plot = plotly_pdp(pdp_result, idx,
                *TestDataExplainer.get_feature_prediction_tuple(idx, col))
    return plot


@app.callback(
    Output('tree-predictions-graph', 'figure'),
    [Input('index-store', 'data')]
)
def update_output_div(idx):
    if idx is not None:
        plot = plotly_tree_predictions(TestDataExplainer.model,
                                        TestDataExplainer.X.iloc[[idx]])
        return plot
    raise PreventUpdate


@app.callback(
    [Output('tree-predictions-table', 'columns'),
     Output('tree-predictions-table', 'data'),],
    [Input('tree-predictions-graph', 'clickData'),
     Input('index-store', 'data')],
    [State('tree-predictions-table', 'columns')])
def display_click_data(clickData, idx, old_columns):
    if clickData is not None:
        model = int(clickData['points'][0]['text'][6:]) if clickData is not None else 0
        shadowtree_df = TestDataExplainer.shadowtree_df(model, idx)
        columns=[{'id': c, 'name': c} for c in  shadowtree_df.columns.tolist()]
        return (columns, shadowtree_df.to_dict('records'))
    raise PreventUpdate


@app.callback(
    [Output('dependence-highlight-index', 'value'),
     Output('dependence-col', 'value'),
     Output('dependence-color-col', 'options'),
     Output('dependence-color-col', 'value')],
    [Input('dependence-shap-scatter-graph', 'clickData')])
def display_scatter_click_data(clickData):
    if clickData is not None:
        #return str(clickData)
        idx = clickData['points'][0]['pointIndex']
        col = clickData['points'][0]['text'].split('=')[0]

        sorted_interact_cols = TestDataExplainer.shap_top_interactions(col)
        dropdown_options = [{'label': col, 'value':col}
                                    for col in sorted_interact_cols]
        return (idx, col, dropdown_options, sorted_interact_cols[1])
    raise PreventUpdate


@app.callback(
    Output('dependence-graph', 'figure'),
    [Input('dependence-col', 'value'),
     Input('dependence-color-col', 'value'),
     Input('dependence-highlight-index', 'value')])
def update_dependence_graph(col, color_col, idx):
    if color_col is not None:
        return plotly_dependence_plot(TestDataExplainer.X,
                TestDataExplainer.shap_values, col, color_col, highlight_idx=idx)
    raise PreventUpdate


@app.callback(
    [Output('interaction-shap-scatter-graph', 'figure'),
     Output('interaction-interact-col', 'options')],
    [Input('interaction-col', 'value')])
def update_interaction_scatter_graph(col):
    interact_cols = TestDataExplainer.shap_top_interactions(col)
    plot = plotly_shap_scatter_plot(
                TestDataExplainer.shap_interaction_values_by_col(col),
                TestDataExplainer.X, interact_cols[:20])
    options = [{'label': col, 'value':col} for col in interact_cols]
    return (plot, options)


@app.callback(
    [Output('interaction-highlight-index', 'value'),
     Output('interaction-interact-col', 'value')],
    [Input('interaction-shap-scatter-graph', 'clickData')])
def display_scatter_click_data(clickData):
    if clickData is not None:
        #return str(clickData)
        idx = clickData['points'][0]['pointIndex']
        col = clickData['points'][0]['text'].split('=')[0]
        return (idx, col)
    raise PreventUpdate


@app.callback(
    Output('interaction-graph', 'figure'),
    [Input('interaction-col', 'value'),
     Input('interaction-interact-col', 'value'),
     Input('interaction-highlight-index', 'value')])
def update_dependence_graph(col, interact_col, idx):
    if interact_col is not None:
        return plotly_dependence_plot(TestDataExplainer.X,
                TestDataExplainer.shap_interaction_values_by_col(col),
                interact_col, col, highlight_idx=idx)
    raise PreventUpdate


if __name__ == '__main__':
    print('Starting server...')
    app.run_server(port=8060)
