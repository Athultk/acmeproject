import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
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
# Error as a difference with squaring it (Works fine)
# def custom_error(y_true, y_pred):
#    return np.sqrt(np.mean((np.min(y_true) - y_pred) ** 2))  # Compute RMSE of the minimum true value and predicted values

# Error as the difference without squaring it (Shows error)
# def custom_error(y_true, y_pred):
#     return np.min(y_true) - y_pred

# Error as the difference without squaring it v1
def custom_error(y_true, y_pred):
    # Calculate the minimum true cost for this fold
    min_cost = np.min(y_true)
    
    # Calculate the mean predicted cost for this fold
    predicted_cost = np.mean(y_pred)
    
    # Calculate the error (difference between minimum true cost and mean predicted cost)
    error = min_cost - predicted_cost
    
    return error  # Return the error for this fold


scorer = make_scorer(custom_error, greater_is_better=False)

# Initialize the model
model = DecisionTreeRegressor()

# Perform Leave-One-Group-Out cross-validation
logo = LeaveOneGroupOut()
scores = cross_val_score(model, X.drop(columns=['Task ID', 'Supplier ID']), y, groups=groups, cv=logo, scoring=scorer)

# Calculate RMSE of the scores
rmse = np.sqrt(np.mean(scores ** 2))
print(f"Cross-Validation RMSE: {rmse}")

# Summarize the results
print(f"Cross-Validation Scores: {scores}")
