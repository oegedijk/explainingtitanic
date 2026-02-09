# xgboost is a dependency of dtreeviz, but too large (>350M) for heroku
# so we uninstall it and mock it here:
from functools import lru_cache
from pathlib import Path
from threading import Lock
from unittest.mock import MagicMock
import sys

from flask import Flask, redirect
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from dash_bootstrap_components.themes import FLATLY
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, RegressionExplainer

from custom import CustomModelTab, CustomPredictionsTab

sys.modules["xgboost"] = MagicMock()

pkl_dir = Path.cwd() / "pkls"

base_app = Flask(__name__)


@base_app.route("/healthz")
def healthz():
    return "ok", 200


@base_app.route("/")
def index():
    return """
    <html>
      <head><title>explainingtitanic</title></head>
      <body>
        <h1>Titanic Explainer Demo</h1>
        <ul>
          <li><a href="/classifier/">Classifier Dashboard</a></li>
          <li><a href="/regression/">Regression Dashboard</a></li>
          <li><a href="/multiclass/">Multiclass Dashboard</a></li>
          <li><a href="/custom/">Custom Dashboard</a></li>
          <li><a href="/simple_classifier/">Simple Classifier Dashboard</a></li>
          <li><a href="/simple_regression/">Simple Regression Dashboard</a></li>
        </ul>
      </body>
    </html>
    """


@base_app.route("/classifier")
def classifier_redirect():
    return redirect("/classifier/", code=302)


@base_app.route("/regression")
def regression_redirect():
    return redirect("/regression/", code=302)


@base_app.route("/multiclass")
def multiclass_redirect():
    return redirect("/multiclass/", code=302)


@base_app.route("/custom")
def custom_redirect():
    return redirect("/custom/", code=302)


@base_app.route("/simple_classifier")
def simple_classifier_redirect():
    return redirect("/simple_classifier/", code=302)


@base_app.route("/simple_regression")
def simple_regression_redirect():
    return redirect("/simple_regression/", code=302)


@lru_cache(maxsize=1)
def _classifier_explainer():
    return ClassifierExplainer.from_file(pkl_dir / "clas_explainer.joblib")


@lru_cache(maxsize=1)
def _regression_explainer():
    return RegressionExplainer.from_file(pkl_dir / "reg_explainer.joblib")


@lru_cache(maxsize=1)
def _multiclass_explainer():
    return ClassifierExplainer.from_file(pkl_dir / "multi_explainer.joblib")


def _build_classifier_app():
    dashboard = ExplainerDashboard(
        _classifier_explainer(),
        title="Classifier Explainer: Predicting survival on the Titanic",
        url_base_pathname="/",
        header_hide_selector=True,
    )
    return dashboard.app.server


def _build_regression_app():
    dashboard = ExplainerDashboard(
        _regression_explainer(),
        title="Regression Explainer: Predicting ticket fare",
        url_base_pathname="/",
    )
    return dashboard.app.server


def _build_multiclass_app():
    dashboard = ExplainerDashboard(
        _multiclass_explainer(),
        title="Multiclass Explainer: Predicting departure port",
        url_base_pathname="/",
    )
    return dashboard.app.server


def _build_custom_app():
    dashboard = ExplainerDashboard(
        _classifier_explainer(),
        [CustomModelTab, CustomPredictionsTab],
        title="Titanic Explainer",
        header_hide_selector=True,
        bootstrap=FLATLY,
        url_base_pathname="/",
    )
    return dashboard.app.server


def _build_simple_classifier_app():
    dashboard = ExplainerDashboard(
        _classifier_explainer(),
        title="Simplified Classifier Dashboard",
        simple=True,
        url_base_pathname="/",
    )
    return dashboard.app.server


def _build_simple_regression_app():
    dashboard = ExplainerDashboard(
        _regression_explainer(),
        title="Simplified Regression Dashboard",
        simple=True,
        url_base_pathname="/",
    )
    return dashboard.app.server


class LazyWSGIApp:
    def __init__(self, factory):
        self._factory = factory
        self._app = None
        self._lock = Lock()

    def __call__(self, environ, start_response):
        if self._app is None:
            with self._lock:
                if self._app is None:
                    self._app = self._factory()
        return self._app(environ, start_response)


app = DispatcherMiddleware(
    base_app,
    {
        "/classifier": LazyWSGIApp(_build_classifier_app),
        "/regression": LazyWSGIApp(_build_regression_app),
        "/multiclass": LazyWSGIApp(_build_multiclass_app),
        "/custom": LazyWSGIApp(_build_custom_app),
        "/simple_classifier": LazyWSGIApp(_build_simple_classifier_app),
        "/simple_regression": LazyWSGIApp(_build_simple_regression_app),
    },
)
