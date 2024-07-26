import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import Ridge

# STEP 5
"""
A ridge regression model which iterates alpha parameter of [0.01, 0.1, 1, 10, 100]

"""

# File paths
scripts_dir = os.path.dirname(__file__)
X_path = os.path.join(scripts_dir, '..', 'data', '5_modeldata', 'X.csv')
y_path = os.path.join(scripts_dir, '..', 'data', '5_modeldata', 'y.csv')
groups_path = os.path.join(scripts_dir, '..', 'data', '5_modeldata', 'groups.csv')

# Read data
X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze("columns")
groups = pd.read_csv(groups_path).squeeze("columns")

# Define the custom scoring function
def custom_scorer(y_true, y_pred, groups):
    # Group by task and find the minimum cost for each task
    min_cost_per_task = y_true.groupby(groups).min()
    # Align the predictions with the true values
    y_pred_aligned = pd.Series(y_pred, index=y_true.index)
    # Calculate the error for each task
    error = y_pred_aligned.groupby(groups).mean() - min_cost_per_task
    # Calculate RMSE
    rmse = np.sqrt((error ** 2).mean())
    return rmse

# This dictionary defines the set of hyperparameters to be tested for the Ridge regression model.
# alpha is the list of regularization strengths to be tested.
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

# Initialize the Ridge regression model
model = Ridge()

# Initialize Leave-One-Group-Out cross-validator
# Same group does not appear in both training and testing sets during cross-validation.
logo = LeaveOneGroupOut()

# Collect results for each parameter setting
best_score = float('inf')
best_params = None
#This loop iterates over each value of alpha in the parameter grid.
for alpha in param_grid['alpha']:
    model.set_params(alpha=alpha)
    scores = []
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_test = groups.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = custom_scorer(y_test, y_pred, groups_test)
        scores.append(score)
    mean_score = np.mean(scores)
    # If the mean score is lower than the best score found so far
    #Update the best score and Update the best parameters with the current alpha
    if mean_score < best_score:
        best_score = mean_score
        best_params = {'alpha': alpha}

print(f"Best parameters: {best_params}")
print(f"Best RMSE: {best_score}")

# STEP 6
"""
A random forest regressor which uses parameters: n_estimators, max_depth, min_samples_split, and 
min_samples_leaf. Please note that this step is taking almost 3 hours to run.

"""

# 6 Hyper-parameter optimization with Random Forrest Model
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from joblib import Parallel, delayed
import joblib
import warnings

warnings.filterwarnings("ignore")

# This custom scoring function calculates the RMSE based on the difference between the predicted costs and the minimum actual costs for each group.
def custom_scorer(y_true, y_pred, groups):
    min_cost_per_task = y_true.groupby(groups).min()
    y_pred_aligned = pd.Series(y_pred, index=y_true.index)
    error = y_pred_aligned.groupby(groups).mean() - min_cost_per_task
    rmse = np.sqrt((error ** 2).mean())
    return rmse

# Wrapper function to make the custom scorer compatible with GridSearchCV
def custom_scorer_grid_search(estimator, X, y, groups):
    y_pred = estimator.predict(X)
    return custom_scorer(y, y_pred, groups)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)

# This dictionary defines the set of hyperparameters to be tested. It includes a smaller set of values to reduce computation time and make the process manageable.
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# This cross-validator ensures that the same task does not appear in both training and testing sets during the cross-validation process.
logo = LeaveOneGroupOut()

# This function evaluates a given set of hyperparameters by performing Leave-One-Group-Out cross-validation. It returns the mean score and the corresponding parameters.
def evaluate_params(params, X, y, groups, logo):
    model.set_params(**params)
    scores = []

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_test = groups.iloc[test_index]

        model.fit(X_train, y_train)
        score = custom_scorer_grid_search(model, X_test, y_test, groups_test)
        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score, params

# This line performs the grid search in parallel using all available CPU cores (n_jobs=-1). 
# The verbose=10 parameter provides progress updates. Each combination of hyperparameters is evaluated by the evaluate_params function.
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate_params)(params, X, y, groups, logo) for params in ParameterGrid(param_grid)
)

# Extract the best parameters and the corresponding score, this line finds the parameter set with the lowest mean score (RMSE).
best_score, best_params = min(results, key=lambda x: x[0])

print(f"Best parameters: {best_params}")
print(f"Best RMSE: {best_score}")

# After finding the best hyperparameters, the model is retrained on the entire dataset with these parameters, and then saved to a file.
model.set_params(**best_params)
model.fit(X, y)
best_model_path = os.path.join(scripts_dir, '..', 'models', 'best_random_forest_model.pkl')
joblib.dump(model, best_model_path)
print(f"Best RandomForest model saved to {best_model_path}")