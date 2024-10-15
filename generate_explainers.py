from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from explainerdashboard import (
    ClassifierExplainer,
    RegressionExplainer,
    ExplainerDashboard,
)
from explainerdashboard.datasets import *

pkl_dir = Path.cwd() / "pkls"

# classifier
print("Generating titanic explainers")
print("Generating classifier explainer")
X_train, y_train, X_test, y_test = titanic_survive()
model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)
clas_explainer = ClassifierExplainer(
    model,
    X_test,
    y_test,
    cats=["Sex", "Deck", "Embarked"],
    descriptions=feature_descriptions,
    labels=["Not survived", "Survived"],
)
_ = ExplainerDashboard(clas_explainer)
clas_explainer.dump(pkl_dir / "clas_explainer.joblib")


# regression
print("Generating titanic fare explainer")
X_train, y_train, X_test, y_test = titanic_fare()
model = RandomForestRegressor(n_estimators=50, max_depth=5).fit(X_train, y_train)
reg_explainer = RegressionExplainer(
    model,
    X_test,
    y_test,
    cats=["Sex", "Deck", "Embarked"],
    descriptions=feature_descriptions,
    units="$",
)
_ = ExplainerDashboard(reg_explainer)
reg_explainer.dump(pkl_dir / "reg_explainer.joblib")

# multiclass
print("Generating titanic embarked multiclass explainer")
X_train, y_train, X_test, y_test = titanic_embarked()
model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, y_train)
multi_explainer = ClassifierExplainer(
    model,
    X_test,
    y_test,
    cats=["Sex", "Deck"],
    descriptions=feature_descriptions,
    labels=["Queenstown", "Southampton", "Cherbourg"],
)
_ = ExplainerDashboard(multi_explainer)
multi_explainer.dump(pkl_dir / "multi_explainer.joblib")
