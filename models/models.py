import os
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# For creation of models

scripts_dir = os.path.dirname(__file__)

models = {
    'ridge_regression': Ridge(random_state=42),
    'random_forest': RandomForestRegressor(random_state=42),
    'gradient_boosting': GradientBoostingRegressor(random_state=42)
}

for model_name, model in models.items():
    model_path = os.path.join(scripts_dir, '..', 'models', f'{model_name}.pkl')
    joblib.dump(model, model_path)