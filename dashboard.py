
# xgboost is a dependency of dtreeviz, but too large (>350M) for heroku
# so we uninstall it and mock it here:
from unittest.mock import MagicMock
import sys
sys.modules["xgboost"] = MagicMock()

from pathlib import Path
from flask import Flask

import dash
from dash_bootstrap_components.themes import FLATLY, BOOTSTRAP # bootstrap theme
from explainerdashboard import *

from index_layout import index_layout, register_callbacks
from custom import CustomModelTab, CustomPredictionsTab

pkl_dir = Path.cwd() / "pkls"

app = Flask(__name__)

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

custom_dashboard = ExplainerDashboard(clas_explainer, 
                        [CustomModelTab, CustomPredictionsTab], 
                        title='Titanic Explainer', header_hide_selector=True,
                        bootstrap=FLATLY,
                        server=app,  url_base_pathname="/custom/")

simple_classifier_dashboard = ExplainerDashboard(clas_explainer,
            title="Simplified Classifier Dashboard", simple=True,
            server=app, url_base_pathname="/simple_classifier/")

simple_regression_dashboard = ExplainerDashboard(reg_explainer,
            title="Simplified Classifier Dashboard", simple=True,
            server=app, url_base_pathname="/simple_regression/")


index_app = dash.Dash(
    __name__, 
    server=app, 
    url_base_pathname="/", 
    external_stylesheets=[BOOTSTRAP])

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

@app.route('/simple_classifier')
def simple_classifier_dashboard():
    return simple_classifier_dashboard.app.index()

@app.route('/simple_regression')
def simple_regression_dashboard():
    return simple_regression_dashboard.app.index()


    
