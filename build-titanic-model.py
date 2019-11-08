from pathlib import Path
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from explainerdashboard.datasets import *

print('loading data...')
X_train, y_train, X_test, y_test = titanic_survive()
train_names, test_names = titanic_names()

print('fitting model...')
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

print('building ExplainerBunch...')
explainer = RandomForestClassifierBunch(model, X_test, y_test, roc_auc_score, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, 
                               labels=['Not survived', 'Survived'])

print('calculating properties...')
explainer.calculate_properties()

print('saving explainer...')
joblib.dump(explainer, Path.cwd() / 'titanic_explainer.joblib')

