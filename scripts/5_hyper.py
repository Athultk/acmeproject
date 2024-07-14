import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import make_scorer

# Define the file paths for the datasets
scripts_dir = os.path.dirname(__file__)
scaled_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
costs_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')
top_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'topsuppliers', 'Top_20_suppliers.csv')

# Load the datasets
tasks_df = pd.read_excel(scaled_tasks_file_path)
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(top_suppliers_file_path)

# Combine the task features, supplier features, and costs into a single dataset
combined_df = costs_df.merge(tasks_df, on='Task ID').merge(suppliers_df, on='Supplier ID')
X = combined_df.drop(columns=['Cost'])
y = combined_df['Cost']
groups = combined_df['Task ID']

# Define the custom scoring function
def custom_error(y_true, y_pred):
    return np.sqrt(np.mean((np.min(y_true) - y_pred) ** 2))  # Compute RMSE of the minimum true value and predicted values

scorer = make_scorer(custom_error, greater_is_better=False)

# Define the hyper-parameter grid for Ridge Regression
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}

# Initialize the model
model = Ridge()

# Perform Grid Search with Leave-One-Group-Out cross-validation
logo = LeaveOneGroupOut()
grid_search = GridSearchCV(model, param_grid, cv=logo, scoring=scorer)
grid_search.fit(X.drop(columns=['Task ID', 'Supplier ID']), y, groups=groups)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Summarize the results
print(f"Grid Search CV Results: {grid_search.cv_results_}")
