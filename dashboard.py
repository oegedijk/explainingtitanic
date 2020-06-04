from sklearn.ensemble import RandomForestClassifier

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
explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, 
                               labels=['Not survived', 'Survived'])

print('Building ExplainerDashboard...')
db = ExplainerDashboard(explainer)

server = db.app.server

if __name__ == '__main__':
    print('Starting server...')
    db.run(8050)
    
    
