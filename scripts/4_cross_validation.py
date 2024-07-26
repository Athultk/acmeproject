import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
import os

'''
This script performs a cross-validation analysis on task cost prediction using a 
RandomForestRegressor. The data is loaded from Excel and CSV files, merged, and preprocessed before 
performing Leave-One-Group-Out cross-validation.
'''

# Load datasets
scripts_dir = os.path.dirname(__file__)
tasks_path = os.path.join(scripts_dir, '..', 'data', '3_scaled', 'tasks_scaled.xlsx')
suppliers_path = os.path.join(scripts_dir, '..', 'data', '4_topsuppliers', 'Top_20_suppliers.csv')
costs_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')

# Load the data into DataFrames
tasks_df = pd.read_excel(tasks_path)
suppliers_df = pd.read_csv(suppliers_path)
costs_df = pd.read_csv(costs_path)

# Merge tasks with costs
merged_df = pd.merge(costs_df, tasks_df, on='Task ID')

# Merge with suppliers
final_df = pd.merge(merged_df, suppliers_df, on='Supplier ID')

# Select relevant columns
final_df = final_df[['Task ID'] + list(tasks_df.columns[1:]) + list(suppliers_df.columns[1:]) + ['Cost']]

# Rename columns for clarity
final_df.columns = final_df.columns.str.replace(' ', '_')

# Separate into X (features) and y (target variable)
X = final_df.drop(['Task_ID', 'Cost'], axis=1)
y = final_df['Cost']
task_ids = final_df['Task_ID']  # Keep track of Task IDs

# Cross validation setup
groups = task_ids
logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)

# Print the splits
for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"Group =\n{groups[test_index]}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

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
task_ids_train = task_ids[train_mask]

# Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Define the scoring function
def score_func(y_true, y_pred):
    error = y_true.min() - y_pred.min()
    return error

# Perform cross-validation
scores = cross_val_score(model, X_train, y_train, groups=task_ids_train, scoring=make_scorer(score_func, greater_is_better=False), cv=logo, n_jobs=6, verbose=0)

print(f'{scores = }')
rmse_score = np.sqrt(np.mean(scores**2))
print(f'{rmse_score = }')
