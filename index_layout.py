import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State

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
                    "the titanic. Showing the full default dashboard."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/classifier"),
                dbc.Button("Show Code", id="clas-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Classifier Dashboard"),
                        dcc.Markdown(
"""
```python

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, feature_descriptions

X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               descriptions=feature_descriptions,
                               labels=['Not survived', 'Survived'])
                               
ExplainerDashboard(explainer).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="clas-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="clas-code-modal",
                    size="lg",
                ),
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
                    "Predicting the fare paid for a ticket on the titanic. "
                    "Showing the full default dashboard."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/regression"),
                dbc.Button("Show Code", id="reg-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Regression Dashboard"),
                        dcc.Markdown(
"""
```python
from sklearn.ensemble import RandomForestRegressor

from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_fare, feature_descriptions

X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = RegressionExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'], 
                                descriptions=feature_descriptions,
                                units="$")
                               
ExplainerDashboard(explainer).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="reg-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="reg-code-modal",
                    size="lg",
                ),
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
                    "Predicting the departure port for passengers on the titanic. "
                    "Showing the full default dashboard."
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/multiclass"),
                dbc.Button("Show Code", id="multi-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Multi Classifier Dashboard"),
                        dcc.Markdown(
"""
```python

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_embarked, feature_descriptions

X_train, y_train, X_test, y_test = titanic_embarked()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck'], 
                                descriptions=feature_descriptions,
                                labels=['Queenstown', 'Southampton', 'Cherbourg'])
                               
ExplainerDashboard(explainer).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="multi-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="multi-code-modal",
                    size="lg",
                ),
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
                html.H4("Customized Classifier Dashboard", className="card-title"),
                html.P(
                    "You can also completely customize the layout and elements of your "
                    "dashboard using a low-code approach."
                    ,className="card-text",
                ),
                # dbc.CardLink("Source code", 
                #     href="https://github.com/oegedijk/explainingtitanic/blob/master/custom.py", 
                #     target="_blank"),
                html.P(),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/custom"),
                dbc.Button("Show Code", id="custom-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Custom Dashboard"),
                        dcc.Markdown(
"""
```python
from explainerdashboard import ExplainerDashboard
from explainerdashboard.custom import *


class CustomModelTab(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Summary")
        self.precision = PrecisionComponent(explainer,
                                title='Precision',
                                hide_subtitle=True, hide_footer=True,
                                hide_selector=True,
                                cutoff=None)
        self.shap_summary = ShapSummaryComponent(explainer,
                                title='Impact',
                                hide_subtitle=True, hide_selector=True,
                                hide_depth=True, depth=8,
                                hide_cats=True, cats=True)
        self.shap_dependence = ShapDependenceComponent(explainer,
                                title='Dependence',
                                hide_subtitle=True, hide_selector=True,
                                hide_cats=True, cats=True,
                                hide_index=True,
                                col='Fare', color_col="PassengerClass")
        self.connector = ShapSummaryDependenceConnector(
                self.shap_summary, self.shap_dependence)

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                     html.H3("Model Performance"),
                    html.Div("As you can see on the right, the model performs quite well."),
                    html.Div("The higher the predicted probability of survival predicted by "
                            "the model on the basis of learning from examples in the training set"
                            ", the higher is the actual percentage of passengers surviving in "
                            "the test set"),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.precision.layout()
                ], style=dict(margin=30))
            ]),
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], style=dict(margin=30)),
                dbc.Col([
                    html.H3("Feature Importances"),
                    html.Div("On the left you can check out for yourself which parameters were the most important."),
                    html.Div(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                    html.Div("If you select 'detailed' you can see the impact of that variable on "
                            "each individual prediction. With 'aggregate' you see the average impact size "
                            "of that variable on the final prediction."),
                    html.Div("With the detailed view you can clearly see that the the large impact from Sex "
                            "stems both from males having a much lower chance of survival and females a much "
                            "higher chance.")
                ], width=4, style=dict(margin=30)),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Feature dependence"),
                    html.Div("In the plot to the right you can see that the higher the cost "
                            "of the fare that passengers paid, the higher the chance of survival. "
                            "Probably the people with more expensive tickets were in higher up cabins, "
                            "and were more likely to make it to a lifeboat."),
                    html.Div("When you color the impacts by PassengerClass, you can clearly see that "
                            "the more expensive tickets were mostly 1st class, and the cheaper tickets "
                            "mostly 3rd class."),
                    html.Div("On the right you can check out for yourself how different features impacted "
                            "the model output."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.shap_dependence.layout()
                ], style=dict(margin=30)),
            ])
        ])

class CustomPredictionsTab(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Predictions")

        self.index = ClassifierRandomIndexComponent(explainer,
                                                    hide_title=True, hide_index=False,
                                                    hide_slider=True, hide_labels=True,
                                                    hide_pred_or_perc=True,
                                                    hide_selector=True, hide_button=False)

        self.contributions = ShapContributionsGraphComponent(explainer,
                                                            hide_title=True, hide_index=True,
                                                            hide_depth=True, hide_sort=True,
                                                            hide_orientation=True, hide_cats=True,
                                                            hide_selector=True,
                                                            sort='importance')

        self.trees = DecisionTreesComponent(explainer,
                                            hide_title=True, hide_index=True,
                                            hide_highlight=True, hide_selector=True)


        self.connector = IndexConnector(self.index, [self.contributions, self.trees])

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Enter name:"),
                    self.index.layout()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Contributions to prediction:"),
                    self.contributions.layout()
                ]),

            ]),
            dbc.Row([

                dbc.Col([
                    html.H3("Every tree in the Random Forest:"),
                    self.trees.layout()
                ]),
            ])
        ])

ExplainerDashboard(explainer, [CustomModelTab, CustomPredictionsTab], 
                        title='Titanic Explainer', header_hide_selector=True,
                        bootstrap=FLATLY).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="custom-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="custom-code-modal",
                    size="xl",
                    scrollable=False
                ),
            ]
        ),
    ],
    style={"width": "18rem"},
)

simple_survive_card = dbc.Card(
    [
        dbc.CardImg(src="assets/titanic.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("Simplified Classifier Dashboard", className="card-title"),
                html.P(
                    "You can generate a simplified single page dashboard "
                    "by passing simple=True to ExplainerDashboard." 
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/simple_classifier"),
                dbc.Button("Show Code", id="simple-clas-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Classifier Dashboard"),
                        dcc.Markdown(
"""
```python

from sklearn.ensemble import RandomForestClassifier

from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, feature_descriptions

X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               descriptions=feature_descriptions,
                               labels=['Not survived', 'Survived'])
                               
ExplainerDashboard(explainer, title="Simplified Classifier Dashboard", simple=True).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="simple-clas-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="simple-clas-code-modal",
                    size="lg",
                ),
            ]
        ),
    ],
    style={"width": "18rem"},
)

simple_ticket_card = dbc.Card(
    [
        dbc.CardImg(src="assets/ticket.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("Simplified Regression Dashboard", className="card-title"),
                html.P(
                    "You can generate a simplified single page dashboard "
                    "by passing simple=True to ExplainerDashboard." 
                    ,className="card-text",
                ),
                html.A(dbc.Button("Go to dashboard", color="primary"),
                       href="/simple_regression"),
                dbc.Button("Show Code", id="simple-reg-code-modal-open", className="mr-1"),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Code needed for this Regression Dashboard"),
                        dcc.Markdown(
"""
```python
from sklearn.ensemble import RandomForestRegressor

from explainerdashboard import RegressionExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_fare, feature_descriptions

X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)

explainer = RegressionExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'], 
                                descriptions=feature_descriptions,
                                units="$")
                               
ExplainerDashboard(explainer, title="Simplified Regression Dashboard", simple=True).run()
```
"""
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="simple-reg-code-modal-close", className="ml-auto")
                        ),
                    ],
                    id="simple-reg-code-modal",
                    size="lg",
                ),
            ]
        ),
    ],
    style={"width": "18rem"},
)

default_cards = dbc.CardDeck([survive_card, ticket_card, port_card])
custom_cards = dbc.CardDeck([simple_survive_card, simple_ticket_card, custom_card])

index_layout =  dbc.Container([
    navbar,     
    dbc.Row([
        dbc.Col([
            html.H3("explainerdashboard"),
            dcc.Markdown("`explainerdashboard` is a python package that makes it easy"
                         " to quickly build an interactive dashboard that explains the inner "
                         "workings of a fitted machine learning model. This allows you to "
                         "open up the 'black box' and show customers, managers, "
                         "stakeholders, regulators (and yourself) exactly how "
                         "the machine learning algorithm generates its predictions."),
            dcc.Markdown("You can explore model performance, feature importances, "
                        "feature contributions (SHAP values), what-if scenarios, "
                        "(partial) dependences, feature interactions, individual predictions, "
                        "permutation importances and even individual decision trees. "
                        "All interactively. All with a minimum amount of code."),
            dcc.Markdown("Works with all scikit-learn compatible models, including XGBoost, Catboost and LightGBM."),
            dcc.Markdown("Due to the modular design, it is also really easy to design your "
                        "own custom dashboards, such as the custom example below."),
        ])
    ], justify="center"),
    dbc.Row([
        dbc.Col([
            html.H3("Installation"),
            dcc.Markdown(
"""
You can install the library with:

```
    pip install explainerdashboard
```

or:

```
    conda install -c conda-forge explainerdashboard
```

""")
        ])
    ], justify="center"),
    
    dbc.Row([
        dbc.Col([
            dcc.Markdown(
"""
More information can be found in the [github repo](http://github.com/oegedijk/explainerdashboard) 
and the documentation on [explainerdashboard.readthedocs.io](http://explainerdashboard.readthedocs.io).
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
        ])
    ], justify="start")
])

def register_callbacks(app):
    @app.callback(
        Output("clas-code-modal", "is_open"),
        Input("clas-code-modal-open", "n_clicks"), 
        Input("clas-code-modal-close", "n_clicks"),
        State("clas-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open

    @app.callback(
        Output("reg-code-modal", "is_open"),
        Input("reg-code-modal-open", "n_clicks"), 
        Input("reg-code-modal-close", "n_clicks"),
        State("reg-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open

    @app.callback(
        Output("multi-code-modal", "is_open"),
        Input("multi-code-modal-open", "n_clicks"), 
        Input("multi-code-modal-close", "n_clicks"),
        State("multi-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open

    @app.callback(
        Output("custom-code-modal", "is_open"),
        Input("custom-code-modal-open", "n_clicks"), 
        Input("custom-code-modal-close", "n_clicks"),
        State("custom-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open

    @app.callback(
        Output("simple-clas-code-modal", "is_open"),
        Input("simple-clas-code-modal-open", "n_clicks"), 
        Input("simple-clas-code-modal-close", "n_clicks"),
        State("simple-clas-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open

    @app.callback(
        Output("simple-reg-code-modal", "is_open"),
        Input("simple-reg-code-modal-open", "n_clicks"), 
        Input("simple-reg-code-modal-close", "n_clicks"),
        State("simple-reg-code-modal", "is_open"),
    )
    def toggle_modal(click_open, click_close, is_open):
        if click_open or click_close:
            return not is_open
        return is_open
