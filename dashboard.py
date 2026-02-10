# xgboost is a dependency of dtreeviz, but too large (>350M) for heroku
# so we uninstall it and mock it here:
from pathlib import Path
from unittest.mock import MagicMock
import sys

from flask import Flask, redirect
from dash import Dash
from dash_bootstrap_components.themes import FLATLY, BOOTSTRAP
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, RegressionExplainer

from custom import CustomModelTab, CustomPredictionsTab
from index_layout import index_layout, register_callbacks

sys.modules["xgboost"] = MagicMock()

pkl_dir = Path.cwd() / "pkls"
app = Flask(__name__)

clas_explainer = ClassifierExplainer.from_file(pkl_dir / "clas_explainer.joblib")
reg_explainer = RegressionExplainer.from_file(pkl_dir / "reg_explainer.joblib")
multi_explainer = ClassifierExplainer.from_file(pkl_dir / "multi_explainer.joblib")

clas_dashboard = ExplainerDashboard(
    clas_explainer,
    title="Classifier Explainer: Predicting survival on the Titanic",
    server=app,
    url_base_pathname="/classifier/",
    header_hide_selector=True,
)

reg_dashboard = ExplainerDashboard(
    reg_explainer,
    title="Regression Explainer: Predicting ticket fare",
    server=app,
    url_base_pathname="/regression/",
)

multi_dashboard = ExplainerDashboard(
    multi_explainer,
    title="Multiclass Explainer: Predicting departure port",
    server=app,
    url_base_pathname="/multiclass/",
)

custom_dashboard = ExplainerDashboard(
    clas_explainer,
    [CustomModelTab, CustomPredictionsTab],
    title="Titanic Explainer",
    header_hide_selector=True,
    bootstrap=FLATLY,
    server=app,
    url_base_pathname="/custom/",
)

simple_classifier_dashboard = ExplainerDashboard(
    clas_explainer,
    title="Simplified Classifier Dashboard",
    simple=True,
    server=app,
    url_base_pathname="/simple_classifier/",
)

simple_regression_dashboard = ExplainerDashboard(
    reg_explainer,
    title="Simplified Regression Dashboard",
    simple=True,
    server=app,
    url_base_pathname="/simple_regression/",
)

index_app = Dash(
    __name__,
    server=app,
    url_base_pathname="/",
    external_stylesheets=[BOOTSTRAP],
)
index_app.title = "explainerdashboard"
index_app.layout = index_layout
register_callbacks(index_app)


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/")
def index():
    return index_app.index()


@app.route("/classifier")
def classifier_redirect():
    return redirect("/classifier/", code=302)


@app.route("/regression")
def regression_redirect():
    return redirect("/regression/", code=302)


@app.route("/multiclass")
def multiclass_redirect():
    return redirect("/multiclass/", code=302)


@app.route("/custom")
def custom_redirect():
    return redirect("/custom/", code=302)


@app.route("/simple_classifier")
def simple_classifier_redirect():
    return redirect("/simple_classifier/", code=302)


@app.route("/simple_regression")
def simple_regression_redirect():
    return redirect("/simple_regression/", code=302)
