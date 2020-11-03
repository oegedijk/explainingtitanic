from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.datasets import *

pkl_dir = Path.cwd() / "pkls"

# classifier
X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)
clas_explainer = ClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               descriptions=feature_descriptions,
                               labels=['Not survived', 'Survived'])
clas_explainer.calculate_properties()
clas_explainer.dump(pkl_dir / "clas_explainer.joblib")


# regression
X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=5).fit(X_train, y_train)
reg_explainer = RegressionExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'], 
                                descriptions=feature_descriptions,
                                units="$")
reg_explainer.calculate_properties()
reg_explainer.dump(pkl_dir / "reg_explainer.joblib")

# multiclass
X_train, y_train, X_test, y_test = titanic_embarked()
model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)
multi_explainer = ClassifierExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck'], 
                                descriptions=feature_descriptions,
                                labels=['Queenstown', 'Southampton', 'Cherbourg'])
multi_explainer.calculate_properties()
multi_explainer.dump(pkl_dir / "multi_explainer.joblib")