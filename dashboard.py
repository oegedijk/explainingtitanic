
# xgboost is a dependency of dtreeviz, but too large (>350M) for heroku
# so we mock it here:
from unittest.mock import MagicMock
import sys
sys.modules["xgboost"] = MagicMock()

from pathlib import Path
from flask import Flask

import dash
import dash_bootstrap_components as dbc
from explainerdashboard import *

from index_layout import index_layout, register_callbacks
from custom import CustomDashboard


app = Flask(__name__)

pkl_dir = Path.cwd() / "pkls"

clas_explainer = ClassifierExplainer.from_file(pkl_dir / "clas_explainer.joblib")
clas_dashboard = ExplainerDashboard(clas_explainer, 
                    title="Classifier Explainer: Predicting survival on the Titanic", 
                    server=app, url_base_pathname="/classifier/", 
                    header_hide_selector=True)

reg_explainer = RegressionExplainer.from_file(pkl_dir / "reg_explainer.joblib")
reg_dashboard = ExplainerDashboard(reg_explainer, 
                    title="Regression Explainer: Predicting ticket fare",
                    server=app, url_base_pathname="/regression/")

multi_explainer = ClassifierExplainer.from_file(pkl_dir / "multi_explainer.joblib")
multi_dashboard = ExplainerDashboard(multi_explainer, 
                    title="Multiclass Explainer: Predicting departure port",
                    server=app, url_base_pathname="/multiclass/")

custom_dashboard = ExplainerDashboard(clas_explainer, CustomDashboard, hide_header=True,
                        server=app,  url_base_pathname="/custom/", 
                        external_stylesheets=[dbc.themes.FLATLY])

index_app = dash.Dash(
    __name__, 
    server=app, 
    url_base_pathname="/", 
    external_stylesheets=[dbc.themes.BOOTSTRAP])

index_app.title = 'explainerdashboard'
index_app.layout = index_layout
register_callbacks(index_app)

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


    
