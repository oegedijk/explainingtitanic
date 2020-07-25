import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("github", href="https://github.com/oegedijk/explainingtitanic"),
            ],
            nav=True,
            in_navbar=True,
            label="Source",
        ),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("github", href="https://github.com/oegedijk/explainerdashboard"),
                dbc.DropdownMenuItem("readthedocs", href="http://explainerdashboard.readthedocs.io/en/latest/"),
                dbc.DropdownMenuItem("pypi", href="https://pypi.org/project/explainerdashboard/"),
            ],
            nav=True,
            in_navbar=True,
            label="explainerdashboard",
        ),
        
    ],
    brand="Titanic Explainer",
    brand_href="https://github.com/oegedijk/explainingtitanic",
    color="primary",
    dark=True,
    fluid=True,
)

survive_card = dbc.Card(
    [
        dbc.CardImg(src="assets/titanic.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("Classifier Dashboard", className="card-title"),
                html.P(
                    "Predicting the probability of surviving "
                    "the titanic."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="http://titanicexplainer.herokuapp.com/classifier"),
            ]
        ),
    ],
    style={"width": "18rem"},
)

ticket_card = dbc.Card(
    [
        dbc.CardImg(src="assets/ticket.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("Regression Dashboard", className="card-title"),
                html.P(
                    "Predicting the fare paid for"
                    " a ticket on the titanic."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="http://titanicexplainer.herokuapp.com/regression"),
            ]
        ),
    ],
    style={"width": "18rem"},
)

port_card = dbc.Card(
    [
        dbc.CardImg(src="assets/port.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("Multiclass Dashboard", className="card-title"),
                html.P(
                    "Predicting the departure port"
                    "for passengers on the titanic."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="http://titanicexplainer.herokuapp.com/multiclass"),
            ]
        ),
    ],
    style={"width": "18rem"},
)

custom_card = dbc.Card(
    [
        dbc.CardImg(src="assets/custom.png", top=True),
        dbc.CardBody(
            [
                html.H4("Custom Dashboard", className="card-title"),
                html.P(
                    "Showing a custom design for a classifier dashboard."
                    ,className="card-text",
                ),
                dbc.CardLink("Source code", href="https://github.com/oegedijk/explainingtitanic/blob/master/custom.py"),
                html.P(),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="http://titanicexplainer.herokuapp.com/custom"),
            ]
        ),
    ],
    style={"width": "18rem"},
)

default_cards = dbc.CardDeck([survive_card, ticket_card, port_card])
custom_cards = dbc.CardDeck([custom_card])

index_layout =  dbc.Container([
    navbar,     
    dbc.Row([
        dbc.Col([
            dcc.Markdown("`explainerdashboard` is a python library for quickly building interactive dashboards " 
             "for explaining the inner workings of machine learning models."),
            dcc.Markdown("The code below is all you need to build and run the classification "
                         "dashboard for example:")
        ])
    ], justify="center"),
    dbc.Row([
        dbc.Col([], width=1),
        dbc.Col([
            dcc.Markdown(
"""
```python

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard.datasets import *
from explainerdashboard.explainers import *

X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, 
                               descriptions=feature_descriptions,
                               labels=['Not survived', 'Survived'])
                               
ExplainerDashboard(explainer).run()
```
"""),
        ], width=8),
        
    ], justify="start"),
    dbc.Row([
        dbc.Col([
            dcc.Markdown("""
Below you can find demonstrations of the three default dashboards for classification, 
regression and multi class classification problems, plus one demonstration of 
a custom dashboard.
"""),
        ])
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            default_cards,
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            custom_cards
        ], width=4)
    ], justify="start")
])