from pathlib import Path
import joblib

from explainerdashboard.dashboards import *

explainer = joblib.load(Path.cwd() / 'titanic_explainer.joblib')

db = RandomForestDashboard(explainer,
                        model_summary=True,
                        contributions=True,
                        shap_dependence=True,
                        shap_interaction=True,
                        shadow_trees=True)

server = db.app.server


if __name__ == '__main__':
    print('Starting server...')
    db.run(8050)