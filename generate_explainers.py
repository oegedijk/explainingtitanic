from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard.datasets import *
from explainerdashboard.explainers import *

from pathlib import Path
import joblib

feature_descriptions = {
    "Sex": "Gender of passenger",
    "Deck": "The deck the passenger had their cabin on",
    "PassengerClass": "The class of the ticket: 1st, 2nd or 3rd class",
    "Fare": "The amount of money people paid", 
    "No_of_relatives_on_board": "number of siblings, spouses, parents plus children on board",
    "Embarked": "the port where the passenger boarded the Titanic. Either Southampton, Cherbourg or Queenstown",
    "Age": "Age of the passenger",
    "No_of_siblings_plus_spouses_on_board": "The sum of the number of siblings plus the number of spouses on board",
    "No_of_parents_plus_children_on_board" : "The sum of the number of parents plus the number of children on board",
}

train_names, test_names = titanic_names()

# classifier
X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
clas_explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                               cats=['Sex', 'Deck', 'Embarked'],
                               idxs=test_names, 
                               descriptions=feature_descriptions,
                               labels=['Not survived', 'Survived'])
clas_explainer.calculate_properties()


# regression
X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X_train, y_train)
reg_explainer = RandomForestRegressionExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck', 'Embarked'], 
                                idxs=test_names, 
                                descriptions=feature_descriptions,
                                units="$")
reg_explainer.calculate_properties()

# multiclass
X_train, y_train, X_test, y_test = titanic_embarked()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
multi_explainer = RandomForestClassifierExplainer(model, X_test, y_test, 
                                cats=['Sex', 'Deck'], 
                                idxs=test_names,
                                descriptions=feature_descriptions,
                                labels=['Queenstown', 'Southampton', 'Cherbourg'])
multi_explainer.calculate_properties()

(Path.cwd() / "pkls").mkdir(parents=True, exist_ok=True)
pkl_dir = Path.cwd() / "pkls"

joblib.dump(clas_explainer, str(pkl_dir / "clas_explainer.pkl"))
joblib.dump(reg_explainer, str(pkl_dir / "reg_explainer.pkl"))
joblib.dump(multi_explainer, str(pkl_dir / "multi_explainer.pkl"))