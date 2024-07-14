import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# STEP 3.1
"""
Combine task features, supplier features, and costs into a single dataset

"""

# Define the file paths for the datasets
scripts_dir = os.path.dirname(__file__)
scaled_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
costs_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')
top_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'topsuppliers', 'Top_20_suppliers.csv')

# Load the datasets
tasks_df = pd.read_excel(scaled_tasks_file_path)
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(top_suppliers_file_path)

# Merge tasks with costs
merged_df = pd.merge(costs_df, tasks_df, on='Task ID')

# Merge with suppliers
final_df = pd.merge(merged_df, suppliers_df, left_on='Supplier ID', right_on='Supplier ID')

# Select relevant columns
final_df = final_df[['Task ID'] + list(tasks_df.columns[1:]) + list(suppliers_df.columns[1:]) + ['Cost']]

# Rename columns for clarity
final_df.columns = final_df.columns.str.replace(' ', '_')

# Separate into X (features) and y (target variable)
X = final_df.drop(['Task_ID', 'Cost'], axis=1)
y = final_df['Cost']
task_ids = final_df['Task_ID']  # Keep track of Task IDs

# STEP 3.2 
'''
Split the dataset into training and testing sets
'''

# Randomly select 20 unique Task ID values for testing
TestGroup = final_df['Task_ID'].sample(20, random_state=42).values

# Split dataset into training and testing sets based on TestGroup
train_mask = ~final_df['Task_ID'].isin(TestGroup)
test_mask = final_df['Task_ID'].isin(TestGroup)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]
task_ids_test = task_ids[test_mask]

# Print shapes of the datasets to verify the split
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# STEP 3.3 
'''
Train the ML model and score its performance
'''

# Train and save multiple models
models = {
    'ridge_regression': Ridge(random_state=42),
    'random_forest': RandomForestRegressor(random_state=42),
    'gradient_boosting': GradientBoostingRegressor(random_state=42)
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{model_name} RMSE on test set: {rmse}")

    # Save the model
    model_path = os.path.join(scripts_dir, '..', 'models', f'{model_name}.pkl')
    joblib.dump(model, model_path)

# STEP 3.4
'''
Define a function to calculate Error (Eq. 1) and using Eq. (2), calculate the RMSE score
'''
# To calculate Error (Eq. 1)
def calculate_error(y_true, y_pred, costs_df, test_task_ids):
    errors = []
    for task_id in test_task_ids:
        # Find the minimum cost for this task_id
        min_cost = costs_df[costs_df['Task ID'] == task_id]['Cost'].min()
        
        # Find the predicted costs for this task_id
        predicted_costs = y_pred[test_task_ids == task_id]  # Filter predictions for this task_id
        
        # Calculate the Error for this task_id (Eq. 1)
        error = min_cost - predicted_costs.mean()
        errors.append(error)
    
    return np.array(errors)

# Load and evaluate each model
model_names = ['ridge_regression', 'random_forest', 'gradient_boosting']
for model_name in model_names:
    # Load the model
    model_path = os.path.join(scripts_dir, '..', 'models', f'{model_name}.pkl')
    model = joblib.load(model_path)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print(f"{model_name} RMSE on test set: {rmse}")
    
    # Calculate Error for each task in TestGroup
    errors = calculate_error(y_test.values, y_pred, costs_df, task_ids_test)
    print(f"Errors for each task in TestGroup: {errors}")

     # Calculate RMSE score (Eq. 2)
    rmse_score = np.sqrt(np.mean(errors**2))
    print(f"{model_name} RMSE score: {rmse_score}")