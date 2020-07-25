from flask import Flask, request, abort, jsonify

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from explainerdashboard.datasets import *

import dash_bootstrap_components as dbc
from custom import CustomDashboard

from pathlib import Path
import joblib

import plotly.io as pio
pio.templates.default = "none"

# feature_descriptions = {
#     "Sex": "Gender of passenger",
#     "Deck": "The deck the passenger had their cabin on",
#     "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
#     "Fare": "The amount of money people paid", 
#     "No_of_relatives_on_board": "number of siblings, spouses, parents plus children on board",
#     "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
#     "Age": "Age of the passenger",
#     "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
#     "No_of_parents_plus_children_on_board" : "The sum of the number of parents plus the number of children on board",
# }

# train_names, test_names = titanic_names()

# # classifier
# X_train, y_train, X_test, y_test = titanic_survive()
# model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
# clas_explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
#                                cats=['Sex', 'Deck', 'Embarked'],
#                                idxs=test_names, 
#                                descriptions=feature_descriptions,
#                                labels=['Not survived', 'Survived'])

# # regression
# X_train, y_train, X_test, y_test = titanic_fare()
# model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)
# reg_explainer = RandomForestRegressionExplainer(model, X_test, y_test, 
#                                 cats=['Sex', 'Deck', 'Embarked'], 
#                                 idxs=test_names, 
#                                 descriptions=feature_descriptions,
#                                 units="$")

# # multiclass
# X_train, y_train, X_test, y_test = titanic_embarked()
# model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
# multi_explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
#                                 cats=['Sex', 'Deck'], 
#                                 idxs=test_names,
#                                 descriptions=feature_descriptions,
#                                 labels=['Queenstown', 'Southampton', 'Cherbourg'])
app = Flask(__name__)

clas_explainer = joblib.load(str(Path.cwd()/"pkls"/"clas_explainer.pkl"))

print('Building ExplainerDashboards...')
clas_dashboard = ExplainerDashboard(clas_explainer, 
                    title="Classifier Explainer: Predicting survival on the Titanic", 
                    server=app, url_base_pathname="/classifier/", 
                    header_hide_selector=True)

# reg_dashboard = ExplainerDashboard(reg_explainer, 
#                     title="Regression Explainer: Predicting ticket fare",
#                     server=app, url_base_pathname="/regression/")

# multi_dashboard = ExplainerDashboard(multi_explainer, 
#                     title="Multiclass Explainer: Predicting departure port",
#                     server=app, url_base_pathname="/multiclass/")

custom_dashboard = ExplainerDashboard(clas_explainer, CustomDashboard, hide_header=True,
                        server=app,  url_base_pathname="/custom/", 
                        external_stylesheets=[dbc.themes.SKETCHY])

@app.route("/")
def index():
    return """
<h1>Explainer Dashboard demonstration</h1>
<p>This is a demonstration of the explainerdashboard package, which allows you to build interactive explanatory dashboard for your machine learning models with just two lines of code.</p>
<p>Two dashboards are hosted at the following urls:</p>
<ol>
<li>The default ExplainerDashboard for binary classifiers, predicting probability of survival on the titanic:
<ul>
<li><a href="classifier/">titanicexplainer.herokuapp.com/classifier</a></li>
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

# <li>The default ExplainerDashboard for regression models, predicting the price of the ticket of passengers: 
# <ul>
# <li><a href="regression/">titanicexplainer.herokuapp.com/regression</a></li>
# </ul>
# </li>
# <li>The default ExplainerDashboard for multiclass classifier models, predicting port of departure:
# <ul>
# <li><a href="multiclass/">titanicexplainer.herokuapp.com/multiclass</a></li>
# </ul>
# </li>

@app.route('/classifier')
def classifier_dashboard():
    return clas_dashboard.app.index()

# @app.route('/regression')
# def regression_dashboard():
#     return reg_dashboard.app.index()

# @app.route('/multiclass')
# def multiclass_dashboard():
#     return multi_dashboard.app.index()

@app.route('/custom')
def custom_dashboard():
    return custom_dashboard.app.index()


    
