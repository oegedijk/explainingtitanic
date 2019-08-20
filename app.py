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

import plotly.io as pio
pio.templates.default = "none"

print("loading DataExplainer object...")

TITLE = 'Titanic Explainer'

explainer = pickle.load(open(
                        Path.cwd() / 'titanic_explainer.pkl', 'rb'))

print('Loading Dash...')
app = dash.Dash(__name__)

app.config['suppress_callback_exceptions']=True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.title = TITLE
server = app.server

print('Defining layout...')

def model_tab(explainer):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2('Model overview:'),
            html.Div([
                html.Label('Bin size:'),
                dcc.Slider(id='precision-binsize',
                            min = 0.01, max = 0.5, step=0.01, value=0.05,
                            marks={0.01: '0.01', 0.05: '0.05', 0.10: '0.10',
                                    0.33: '0.33', 0.5: '0.5'}, included=False),
            ], style={'margin': 20}),
            dcc.Graph(id='precision-graph'),
            html.Div([
                html.Label('Cutoff:'),
                dcc.Slider(id='precision-cutoff',
                            min = 0.01, max = 0.99, step=0.01, value=0.5,
                            marks={0.01: '0.01', 0.25: '0.25', 0.50: '0.50',
                                    0.75: '0.75', 0.99: '0.99'}, included=False),
            ], style={'margin': 20}),
        ]),
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
                        min = 1, max = len(explainer.columns),
                        value=15),
            dcc.Graph(id='importances-graph'),
        ])
    ]),
    ], fluid=True)


def contributions_tab(explainer):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Label('Fill specific index'),
            dbc.Input(id='input-index', placeholder="Fill in index here...",
                        debounce=True),
            html.Div([
                dcc.RangeSlider(
                    id='prediction-range-slider',
                    min=0.0, max=1.0, step=0.01,
                    value=[0.5, 1.0],  allowCross=False,
                    marks={0.0:'0.0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3',
                            0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7',
                            0.8:'0.8', 0.9:'0.9', 1.0:'1.0'})
            ], style={'margin': 20}),
            html.Div([
                dcc.RadioItems(
                    id='include-labels',
                    options=[
                        {'label': explainer.labels[1], 'value': 'pos'},
                        {'label': explainer.labels[0], 'value': 'neg'},
                        {'label': 'Both/either', 'value': 'any'},
                    ],
                    value='any',
                    labelStyle={'display': 'inline-block'}
                )
            ], style={'margin': 20}),
            html.Div([
                html.Button('random index', id='index-button'),
            ], style={'margin': 30})
        ], width=4),
        dbc.Col([
             dcc.Loading(id="loading-model-prediction",
                         children=[dcc.Markdown(id='model-prediction')]),
        ]),
        dcc.Store(id='index-store'),
    ], justify="between"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label('Number of features to display:'),
                dcc.Slider(id='contributions-size',
                    min = 1, max = len(explainer.columns),
                    marks={int(i) : str(int(i))
                                for i in np.linspace(
                                        1, len(explainer.columns), 6)},
                    step = 1, value=15),
            ]),
            html.Div(id='contributions-size-display', style={'margin-top': 20})
        ], width=4),

    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3('Contributions to prediction'),
                html.Label('(click on a bar to display pdp graph)'),
                dcc.Loading(id="loading-contributions-graph",
                         children=[dcc.Graph(id='contributions-graph')])
            ], style={'margin': 30}),
            html.Div(id='contributions-clickdata')

        ], width=6),
        dbc.Col([
            html.Div([
                html.H3('Partial Dependence Plot'),
                html.Label("Plot partial dependence plot (\'what if?\') for column:"),
                dcc.Dropdown(id='pdp-col',
                    options=[{'label': col, 'value':col}
                                for col in explainer.mean_abs_shap_df(cats=True)\
                                                            .Feature.tolist()],
                    value=explainer.mean_abs_shap_df(cats=True).Feature[0]),
                dcc.Loading(id="loading-pdp-graph",
                        children=[dcc.Graph(id='pdp-graph')]),
            ], style={'margin': 30})
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('Contributions to prediction'),
            html.Label('(table format)'),
            dash_table.DataTable(
                id='contributions_table',
                columns=[{'id': c, 'name': c}
                            for c in ['Reason', 'Effect']],
            ),

        ], width=6),
    ]),
    ], fluid=True)


def dependence_tab(explainer):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Individual Shap Values'),
            html.Label('(Click on a dot to display dependence graph)'),
            dcc.Graph(id='dependence-shap-scatter-graph',
                      figure=explainer.plot_shap_summary(topx=20))
        ]),
        dbc.Col([
            html.Label('Highlight index:'),
            dbc.Input(id='dependence-highlight-index',
                        placeholder="Highlight index...",
                        debounce=True),
            html.Label('Plot dependence for column:'),
            dcc.Dropdown(id='dependence-col',
                    options=[{'label': col, 'value':col}
                                for col in explainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=explainer.mean_abs_shap_df().Feature[0]),
            html.Label('Color observation by column:'),
            dcc.Dropdown(id='dependence-color-col',
                    options=[{'label': col, 'value':col}
                                for col in explainer.columns]),
            dcc.Graph(id='dependence-graph')
        ])
    ]),
    ] ,  fluid=True)


def interactions_tab(explainer):
    return dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3('Shap Interaction Values'),
            html.Label('Display shap interaction values for column:'),
            dcc.Dropdown(id='interaction-col',
                    options=[{'label': col, 'value':col}
                                for col in explainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=explainer.mean_abs_shap_df().Feature[0]),
            html.Label('(Click on a dot to display interaction graph)'),
            dcc.Loading(id="loading-interaction-shap-scatter",
                         children=[dcc.Graph(id='interaction-shap-scatter-graph')])
        ], width=6),
        dbc.Col([
            html.Label('Highlight index:'),
            dbc.Input(id='interaction-highlight-index',
                        placeholder="Highlight index...", debounce=True),
            html.Label('Show interaction with column:'),
            dcc.Dropdown(id='interaction-interact-col',
                    options=[{'label': col, 'value':col}
                                for col in explainer.mean_abs_shap_df()\
                                                            .Feature.tolist()],
                    value=explainer.mean_abs_shap_df().Feature[0]),
            html.Label('Shap interaction values:'),
            dcc.Loading(id="loading-interaction-graph",
                         children=[dcc.Graph(id='interaction-graph')]),
        ], width=6)
    ]),
    ],  fluid=True)


def trees_tab(explainer):
    return dbc.Container([
     dbc.Row([
        dbc.Col([
            html.H2('Predictions of individual decision trees.'),
            html.H4('(click on a prediction to see decision path)'),
            dcc.Graph(id='tree-predictions-graph'),
        ], width={"size": 8, "offset": 2})
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Decision path in decision tree:'),
            dcc.Markdown(id='tree-basevalue'),
            dash_table.DataTable(
                id='tree-predictions-table',
            ),
        ], width={"size": 6, "offset": 3})
    ]),
    ],  fluid=True)


app.layout = dbc.Container([
    dbc.Row([html.H1(TITLE)]),
    dcc.Tabs(
        [
            dcc.Tab(children=model_tab(explainer),
                    label='Model Overview',
                    id='model_tab'),
            dcc.Tab(children=dependence_tab(explainer),
                    label='Dependence Plots',
                    id='dependence_tab'),
            dcc.Tab(children=interactions_tab(explainer),
                    label='Interactions graphs',
                    id='interactions_tab'),
            dcc.Tab(children=contributions_tab(explainer),
                    label='Individual Contributions',
                    id='contributions_tab'),
            dcc.Tab(children=trees_tab(explainer),
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
    return explainer.plot_precision(bin_size, cutoff)


@app.callback(
    [Output('confusionmatrix-graph', 'figure'),
     Output('roc-auc-graph', 'figure'),
     Output('pr-auc-graph', 'figure')],
    [Input('precision-cutoff', 'value'),
     Input('confusionmatrix-normalize', 'value')],
)
def update_precision_graph(cutoff, normalized):
    confmat_plot = explainer.plot_confusion_matrix(
                        cutoff=cutoff, normalized=normalized=='Normalized')
    roc_auc_plot = explainer.plot_roc_auc(cutoff=cutoff)
    pr_auc_plot = explainer.plot_pr_auc(cutoff=cutoff)
    return (confmat_plot, roc_auc_plot, pr_auc_plot)


@app.callback(
     Output('importances-graph', 'figure'),
    [Input('importance-tablesize', 'value'),
     Input('group-categoricals', 'value'),
     Input('permutation-or-shap', 'value')]
)
def update_importances(tablesize, cats, permutation_shap):
    return explainer.plot_importances(
                type=permutation_shap, topx=tablesize, cats=cats)


@app.callback(
    Output('input-index', 'value'),
    [Input('index-button', 'n_clicks')],
    [State('prediction-range-slider', 'value'),
     State('include-labels', 'value')]
)
def update_input_index(n_clicks, slider_range, include):
    y = None
    if include=='neg': y = 0
    elif include=='pos': y = 1
    idx = explainer.random_index(
            y=y, pred_proba_min=slider_range[0], pred_proba_max=slider_range[1])
    if idx is not None:
        return idx
    raise PreventUpdate


@app.callback(
    Output('index-store', 'data'),
    [Input('input-index', 'value')],
    [State('index-store', 'data')]
)
def update_bsn_div(input_index, old_index):
    if str(input_index).isdigit() and int(input_index) <= len(explainer):
        return int(input_index)
    raise PreventUpdate


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
    model_prediction = f"##### Index: {idx}\n"\
                        + f"## Prediction: {np.round(100*explainer.pred_probas[idx],2)}% {explainer.labels[1]}\n"\
                        + f"## Actual Outcome: {explainer.labels[explainer.y[idx]]}"
    plot = explainer.plot_contributions(idx, topx=topx)
    summary_table = explainer.contrib_summary_df(idx, topx=topx).to_dict('records')
    return (model_prediction, plot, summary_table)


@app.callback(
    Output('pdp-col', 'value'),
    [Input('contributions-graph', 'clickData')])
def update_pdp_col(clickData):
    if clickData is not None:
        col = clickData['points'][0]['x']
        return col
    raise PreventUpdate


@app.callback(
    Output('pdp-graph', 'figure'),
    [Input('index-store', 'data'),
     Input('pdp-col', 'value')]
)
def update_pdp_graph(idx, col):
    return explainer.plot_pdp(idx, col)


@app.callback(
    Output('tree-predictions-graph', 'figure'),
    [Input('index-store', 'data')]
)
def update_output_div(idx):
    if idx is not None:
        return explainer.plot_trees(idx)
    raise PreventUpdate


@app.callback(
    [Output('tree-basevalue', 'children'),
     Output('tree-predictions-table', 'columns'),
     Output('tree-predictions-table', 'data'),],
    [Input('tree-predictions-graph', 'clickData'),
     Input('index-store', 'data')],
    [State('tree-predictions-table', 'columns')])
def display_click_data(clickData, idx, old_columns):
    if clickData is not None:
        model = int(clickData['points'][0]['text'][6:]) if clickData is not None else 0
        (baseval, prediction,
                shadowtree_df) = explainer.shadowtree_df_summary(model, idx)
        columns=[{'id': c, 'name': c} for c in  shadowtree_df.columns.tolist()]
        baseval_str = f"Tree no {model}, Starting prediction   : {baseval}, final prediction : {prediction}"
        return (baseval_str, columns, shadowtree_df.to_dict('records'))
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

        sorted_interact_cols = explainer.shap_top_interactions(col)
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
        return explainer.plot_dependence(
                    col, color_col, highlight_idx=idx)
    raise PreventUpdate


@app.callback(
    [Output('interaction-shap-scatter-graph', 'figure'),
     Output('interaction-interact-col', 'options')],
    [Input('interaction-col', 'value')])
def update_interaction_scatter_graph(col):
    plot = explainer.plot_shap_interaction_summary(col, topx=20)
    interact_cols = explainer.shap_top_interactions(col)
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
        return explainer.plot_interaction_dependence(
                col, interact_col, highlight_idx=idx)
    raise PreventUpdate


if __name__ == '__main__':
    print('Starting server...')
    app.run_server(port=8065)
