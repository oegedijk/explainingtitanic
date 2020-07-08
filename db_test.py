from flask import Flask, request, abort, jsonify

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from explainerdashboard.datasets import *

import dash_bootstrap_components as dbc
from custom import CustomDashboard

import plotly.io as pio
pio.templates.default = "none"

print('loading data...')
X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()

print('fitting model...')
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

print('building ExplainerBunch...')
explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, 
                               labels=['Not survived', 'Survived'])


app = Flask(__name__)

print('Building ExplainerDashboard...')
default_dashboard = ExplainerDashboard(explainer, server=app, url_base_pathname="/default/")
custom_dashboard = ExplainerDashboard(explainer, CustomDashboard, 
                        server=app,  url_base_pathname="/custom/", 
                        external_stylesheets=[dbc.themes.SKETCHY])

@app.route("/")
def index():
    return """
<h1>Explainer Dashboard demonstration</h1>
<p>This is a demonstration of the explainerdashboard package, which allows you to build interactive explanatory dashboard for your machine learning models with just two lines of code.</p>
<p>Two dashboards are hosted at the following urls:</p>
<ol>
<li>The default ExplainerDashboard including every tab and every ExplainerComponent:
<ul>
<li><a href="default/">titanicexplainer.herokuapp.com/default</a></li>
</ul>
</li>
<li>A custom dashboard showcasing how you combine ExplainerComponents together with your own layout and styling:
<ul>
<li><a href="custom/">titanicexplainer.herokuapp.com/custom</a></li>
</ul>
</li>
</ol>
<h2>Source</h2>
<p>This demonstration is hosted at <a href="http://github.com/oegedijk/explainingtitanic">http://github.com/oegedijk/explainingtitanic</a></p>
<h2>ExplainerDashboard</h2>
<p>Github: <a href="http://github.com/oegedijk/explainerdashboard">http://github.com/oegedijk/explainerdashboard</a></p>
<p>Documentation: <a href="explainerdashboard.readthedocs.io">explainerdashboard.readthedocs.io</a></p>
"""

@app.route('/default')
def default_dashboard():
    return db1.app.index()

@app.route('/custom')
def custom_dashboard():
    return db2.app.index()

