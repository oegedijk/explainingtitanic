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
                    "Predicting the departure port "
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
                dbc.CardLink("Source code", 
                    href="https://github.com/oegedijk/explainingtitanic/blob/master/custom.py", 
                    target="_blank"),
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
            html.H3("explainerdashboard"),
            dcc.Markdown("`explainerdashboard` is a python package that makes it easy"
                         " to quickly build an interactive dashboard that explains the inner "
                         "workings of a machine learning model."),
            dcc.Markdown("This allows you to open up the 'black box' and show "
                        "customers, managers, stakeholders, regulators (and yourself) "
                        "exactly how the machine learning algorithmn generates its predictions."),
            dcc.Markdown("You can explore model performance, feature importances, "
                        "feature contributions (SHAP values).", 
                         "(partial) dependences, individual predictions, "
                        "permutation importances and even individual decision trees "
                        "within a random forest. All interactively. All with a minimum amount of code."),
            dcc.Markdown("Works with all scikit-learn compatible models, including XGBoost, Catboost and LightGBM."),
            dcc.Markdown("Due to the modular design, it is also really easy to design your "
                        "own custom dashboards, such as the custom example below."),
            dcc.Markdown("Click on 'show code' to see all the code needed to build and "
                         "run the classifier dashboard example below.")
        ])
    ], justify="center"),
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Show code",
                id="collapse-button",
                className="mb-3",
                color="primary",
            ),
            dbc.Collapse(
                html.Div([
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
"""),]),
                
                id="collapse",
            ),
            
        ], width=8),
        
    ], justify="start"),
    dbc.Row([
        dbc.Col([
            html.H3("Installation"),
            dcc.Markdown(
"""
You can install the library with:

```
    pip install explainerdashboard
```

""")
        ])
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            dcc.Markdown(
"""
More information can be found in the [github repo](http://github.com/oegedijk/explainerdashboard) 
and the documentation on [readthedocs.io](http://explainerdashboard.readthedocs.io).
""")
        ])
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Examples"),
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
