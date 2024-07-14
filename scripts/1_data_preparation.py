import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

# STEP 1.1
'''
Verify that there are no missing values, that the IDs in all data sets provided match, and count the number 
of tasks, suppliers, features and cost values in all data sets. Remove tasks which are not having any costs.
''' 

# Load the datasets
scripts_dir = os.path.dirname(__file__)

# Construct the path to the files
tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'formatted', 'tasks_formatted.xlsx')
costs_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')
suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'suppliers.csv')

# Read the files
tasks_df = pd.read_excel(tasks_file_path)
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(suppliers_file_path)

# Verify that there are no missing values
print("Missing values in tasks dataset:\n", tasks_df.isnull().sum())
print("Missing values in suppliers dataset:\n", suppliers_df.isnull().sum())
print("Missing values in costs dataset:\n", costs_df.isnull().sum())

# Ensure IDs match
task_ids = set(tasks_df['Task ID'])
supplier_ids = set(suppliers_df['Supplier ID'])
cost_task_ids = set(costs_df['Task ID'])
cost_supplier_ids = set(costs_df['Supplier ID'])

print("Number of unique task IDs in tasks dataset:", len(task_ids))
print("Number of unique supplier IDs in suppliers dataset:", len(supplier_ids))
print("Number of unique task IDs in costs dataset:", len(cost_task_ids))
print("Number of unique supplier IDs in costs dataset:", len(cost_supplier_ids))

# Check for mismatched IDs
mismatched_task_ids = task_ids - cost_task_ids
mismatched_supplier_ids = supplier_ids - cost_supplier_ids

print("Mismatched task IDs:", mismatched_task_ids)
print("Mismatched supplier IDs:", mismatched_supplier_ids)

# Count the number of tasks, suppliers, features, and cost values
num_tasks = len(tasks_df)
num_suppliers = len(suppliers_df)
num_task_features = tasks_df.shape[1] - 1  # Subtracting the Task ID column
num_supplier_features = suppliers_df.shape[1] - 1  # Subtracting the Supplier ID column
num_cost_values = costs_df.shape[0]

print("Number of tasks:", num_tasks)
print("Number of suppliers:", num_suppliers)
print("Number of task features:", num_task_features)
print("Number of supplier features:", num_supplier_features)
print("Number of cost values:", num_cost_values)

# Remove tasks with missing cost values from datasets
tasks_with_costs = tasks_df[tasks_df['Task ID'].isin(cost_task_ids)]
tasks_removed = tasks_df[~tasks_df['Task ID'].isin(cost_task_ids)]

print("Number of tasks removed due to missing cost values:", len(tasks_removed))

# Construct the path to save the filtered tasks file
filtered_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'filtered', 'tasks_filtered.xlsx')

# Save the cleaned datasets if necessary
tasks_with_costs.to_excel(filtered_tasks_file_path, index=False)

print("Removed tasks which are not having any costs and saved as tasks_filtered in filtered folder")

# STEP 1.2
'''
Calculate the maximum, minimum, mean and variance of each feature. Some features have very low variance 
(less than 0.01) and will not be useful for learning. Identify and remove those columns from the data.
''' 

# Read the filtered task file
tasks_df = pd.read_excel(filtered_tasks_file_path)

# Remove the Task ID and Supplier ID columns before calculations
tasks_features = tasks_df.drop(columns=['Task ID'])
suppliers_features = suppliers_df.drop(columns=['Supplier ID'])

# Function to print the statistics and remove low variance columns
def process_features(df, name):
    print(f"Statistics for {name} dataset:")
    stats = df.describe().transpose()
    variance = df.var()
    stats['variance'] = variance
    
    print("Maximum:\n", stats['max'])
    print("Minimum:\n", stats['min'])
    print("Mean:\n", stats['mean'])
    print("Variance:\n", stats['variance'])
    
    # Identify features with variance less than 0.01
    low_variance_features = variance[variance < 0.01].index
    print(f"\nColumns with variance less than 0.01 in {name} dataset: {list(low_variance_features)}\n")
    
    # Remove low variance columns
    df_reduced = df.drop(columns=low_variance_features)
    
    print(f"{name} dataset reduced from {df.shape[1]} to {df_reduced.shape[1]} columns.\n")
    return df_reduced

# Process the features and remove low variance columns
tasks_features_reduced = process_features(tasks_features, 'tasks')
suppliers_features_reduced = process_features(suppliers_features, 'suppliers')

# Add back the Task ID and Supplier ID columns to the reduced datasets
tasks_df_reduced = pd.concat([tasks_df['Task ID'], tasks_features_reduced], axis=1)
suppliers_df_reduced = pd.concat([suppliers_df['Supplier ID'], suppliers_features_reduced], axis=1)

# Construct the path to save the reduced tasks and suppliers files
reduced_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'reduced', 'tasks_reduced.xlsx')
reduced_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'reduced', 'suppliers_reduced.csv')

# Save the updated tasks and suppliers to an Excel file
tasks_df_reduced.to_excel(reduced_tasks_file_path, index=False)
suppliers_df_reduced.to_csv(reduced_suppliers_file_path, index=False)

print("Features with low variance are removed from tasks and suppliers and are saved in reduced folder.")

# STEP 1.3
'''
Features have very different ranges. Scale all features to the range [-1,1] (not the IDs nor the cost!).
''' 

from sklearn.preprocessing import MinMaxScaler

# Read the reduced tasks and suppliers files
tasks_df = pd.read_excel(reduced_tasks_file_path)
suppliers_df = pd.read_csv(reduced_suppliers_file_path)

# Separate the ID columns and the features
task_ids = tasks_df['Task ID']
supplier_ids = suppliers_df['Supplier ID']
task_features = tasks_df.drop(columns=['Task ID'])
supplier_features = suppliers_df.drop(columns=['Supplier ID'])

# Initialize the MinMaxScaler to scale features to the range [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))

# Scale the features
task_features_scaled = scaler.fit_transform(task_features)
supplier_features_scaled = scaler.fit_transform(supplier_features)

# Convert the scaled features back to DataFrame
task_features_scaled_df = pd.DataFrame(task_features_scaled, columns=task_features.columns)
supplier_features_scaled_df = pd.DataFrame(supplier_features_scaled, columns=supplier_features.columns)

# Concatenate the ID columns back with the scaled features
tasks_df_scaled = pd.concat([task_ids, task_features_scaled_df], axis=1)
suppliers_df_scaled = pd.concat([supplier_ids, supplier_features_scaled_df], axis=1)

# Construct the path to save the scaled tasks and suppliers files
scaled_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
scaled_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'suppliers_scaled.csv')

# Save the scaled datasets
tasks_df_scaled.to_excel(scaled_tasks_file_path, index=False)
suppliers_df_scaled.to_csv(scaled_suppliers_file_path, index=False)

print("Features scaled to the range [-1, 1] and scaled datasets saved in scaled folder.")

# STEP 1.4
'''
For tasks only, calculate the absolute correlation of all pairs of features and visualize it. One feature at a time, 
identify which feature is the most correlated with one or more of the others (absolute correlation larger than or 
equal to 0.8), remove it from the dataset, recalculate the correlations and repeat until all pairs of features 
have an absolute correlation smaller than 0.8.
''' 

from PIL import Image

# Read the scaled task file
scaled_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
tasks_df = pd.read_excel(scaled_tasks_file_path)

# Separate the ID column and the features
task_ids = tasks_df['Task ID']
task_features = tasks_df.drop(columns=['Task ID'])

# Function to calculate the absolute correlation matrix
def calculate_absolute_correlation(features):
    corr_matrix = features.corr().abs()
    return corr_matrix

# Function to visualize the correlation matrix
def visualize_correlation(corr_matrix, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    
    # Display the image
    image = Image.open(save_path)
    image.show()

# Function to remove highly correlated features
def remove_highly_correlated_features(features, threshold=0.8):
    corr_matrix = calculate_absolute_correlation(features)
    # Initialize the set of features to be removed
    features_to_remove = set()

    while True:
        # Find pairs of features with absolute correlation above the threshold
        correlated_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns 
                            if i != j and corr_matrix.loc[i, j] >= threshold]
        
        if not correlated_pairs:
            break
        
        # Find the feature with the most correlations
        most_correlated_feature = None
        max_correlations = 0
        for feature in corr_matrix.columns:
            num_correlations = sum(corr_matrix.loc[feature, :] >= threshold) - 1
            if num_correlations > max_correlations:
                max_correlations = num_correlations
                most_correlated_feature = feature

        # Remove the most correlated feature
        if most_correlated_feature:
            features_to_remove.add(most_correlated_feature)
            corr_matrix.drop(index=most_correlated_feature, columns=most_correlated_feature, inplace=True)
    
    # Remove the identified features from the dataset
    reduced_features = features.drop(columns=features_to_remove)
    return reduced_features, features_to_remove

# Define the paths where the images will be saved
scripts_dir = os.path.dirname(__file__)
initial_corr_image_path = os.path.join(scripts_dir, '..', 'reports', 'figures', 'initial_correlation_matrix.png')
final_corr_image_path = os.path.join(scripts_dir, '..', 'reports', 'figures', 'final_correlation_matrix.png')

# Calculate the initial absolute correlation matrix and visualize it
initial_corr_matrix = calculate_absolute_correlation(task_features)
visualize_correlation(initial_corr_matrix, "Initial Correlation Matrix", initial_corr_image_path)

# Remove highly correlated features
reduced_task_features, removed_features = remove_highly_correlated_features(task_features)

# Calculate the final absolute correlation matrix and visualize it
final_corr_matrix = calculate_absolute_correlation(reduced_task_features)
visualize_correlation(final_corr_matrix, "Final Correlation Matrix", final_corr_image_path)

print("Initial and Final correlation matrixes are saved.")

# STEP 1.5
'''
Identify the top 20 suppliers for each task (the ones with the lowest cost). Remove from the data all suppliers 
that never appear in the top 20 of any task.
''' 

# Load the datasets
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(scaled_suppliers_file_path)

# Initialize a set to keep track of suppliers that appear in the top 20 for any task
top_suppliers = set()

# Identify the top 20 suppliers for each task
for task_id in costs_df['Task ID'].unique():
    # Filter costs for the current task
    task_costs = costs_df[costs_df['Task ID'] == task_id]
    # Identify the top 20 suppliers with the lowest costs
    top_20_suppliers = task_costs.nsmallest(20, 'Cost')['Supplier ID']
    # Add these suppliers to the set of top suppliers
    top_suppliers.update(top_20_suppliers)

# Filter the suppliers dataset to keep only the top suppliers
top_suppliers_df = suppliers_df[suppliers_df['Supplier ID'].isin(top_suppliers)]

# Save the filtered suppliers dataset
top_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'topsuppliers', 'Top_20_suppliers.csv')
top_suppliers_df.to_csv(top_suppliers_file_path, index=False)

print("Top 20 suppliers dataset saved.")
print("Number of suppliers retained:", len(top_suppliers_df))