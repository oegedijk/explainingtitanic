from flask import Flask, request, abort, jsonify

from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from custom import CustomDashboard
from index_layout import index_layout

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from pathlib import Path
import joblib

import plotly.io as pio
pio.templates.default = "none"

app = Flask(__name__)

clas_explainer = joblib.load(str(Path.cwd()/"pkls"/"clas_explainer.pkl"))
reg_explainer = joblib.load(str(Path.cwd()/"pkls"/"reg_explainer.pkl"))
multi_explainer = joblib.load(str(Path.cwd()/"pkls"/"multi_explainer.pkl"))

print('Building ExplainerDashboards...')
clas_dashboard = ExplainerDashboard(clas_explainer, 
                    title="Classifier Explainer: Predicting survival on the Titanic", 
                    server=app, url_base_pathname="/classifier/", 
                    header_hide_selector=True)

reg_dashboard = ExplainerDashboard(reg_explainer, 
                    title="Regression Explainer: Predicting ticket fare",
                    server=app, url_base_pathname="/regression/")

multi_dashboard = ExplainerDashboard(multi_explainer, 
                    title="Multiclass Explainer: Predicting departure port",
                    server=app, url_base_pathname="/multiclass/")

custom_dashboard = ExplainerDashboard(clas_explainer, CustomDashboard, hide_header=True,
                        server=app,  url_base_pathname="/custom/", 
                        external_stylesheets=[dbc.themes.FLATLY])

index_app = dash.Dash(__name__, server=app, url_base_pathname="/", external_stylesheets=[dbc.themes.BOOTSTRAP])
index_app.layout = index_layout

@index_app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    print("triggered")
    if n:
        return not is_open
    return is_open

@app.route("/")
def index():
    return index_app.index()

@app.route('/classifier')
def classifier_dashboard():
    return clas_dashboard.app.index()

@app.route('/regression')
def regression_dashboard():
    return reg_dashboard.app.index()

@app.route('/multiclass')
def multiclass_dashboard():
    return multi_dashboard.app.index()

@app.route('/custom')
def custom_dashboard():
    return custom_dashboard.app.index()


    
