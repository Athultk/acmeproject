import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

# Set the Matplotlib backend to 'Agg' for compatibility with non-interactive environments
import matplotlib
matplotlib.use('Agg')

# Define the file paths for the datasets
scripts_dir = os.path.dirname(__file__)
scaled_tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
costs_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')
top_suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'topsuppliers', 'Top_20_suppliers.csv')

# Load the datasets
tasks_df = pd.read_excel(scaled_tasks_file_path)
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(top_suppliers_file_path)

# STEP 2.1
'''
Multiple Figures for Boxplots of Task Features
To create one figure (using boxplots) that shows the distribution of feature values for eachtask
'''

# Remove the Task ID column for plotting
tasks_features_df = tasks_df.drop(columns=['Task ID'])

# Number of features per figure
features_per_figure = 21
num_features = tasks_features_df.shape[1]
num_figures = math.ceil(num_features / features_per_figure)

# Define the directory to save the figures
save_dir = os.path.join(scripts_dir, '..', 'reports', 'figures')

# Create and save each figure
for i in range(num_figures):
    start_col = i * features_per_figure
    end_col = min((i + 1) * features_per_figure, num_features)
    
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=tasks_features_df.iloc[:, start_col:end_col], orient='h')
    plt.title(f'Distribution of Task Feature Values (Features {start_col + 1} to {end_col})')
    plt.xlabel('Value')
    plt.ylabel('Task Features')
    plt.tight_layout()
    
    # Save each figure with a unique filename
    save_path = os.path.join(save_dir, f'task_features_distribution_{i + 1}.png')
    plt.savefig(save_path)
    plt.close()

print("Boxplots of Task Features are saved")

# STEP 2.2
'''
Distribution of Errors for Each Supplier with RMSE Annotation
'''

# Compute the error for each supplier and task
costs_pivot = costs_df.pivot(index='Task ID', columns='Supplier ID', values='Cost')

min_costs = costs_pivot.min(axis=1)
errors_df = costs_pivot.sub(min_costs, axis=0)

# Calculate RMSE for each supplier
rmse_values = np.sqrt((errors_df ** 2).mean(axis=0))

# Number of suppliers per figure
suppliers_per_figure = 22
num_suppliers = errors_df.shape[1]
num_supplier_figures = math.ceil(num_suppliers / suppliers_per_figure)

# Create and save each figure
for i in range(num_supplier_figures):
    start_col = i * suppliers_per_figure
    end_col = min((i + 1) * suppliers_per_figure, num_suppliers)
    
    # Select the subset of data for the current suppliers
    subset_errors_df = errors_df.iloc[:, start_col:end_col]

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=subset_errors_df, orient='h')

    # Annotate each boxplot with the RMSE value
    for j, supplier in enumerate(subset_errors_df.columns):
        plt.text(subset_errors_df[supplier].max() + 0.01, j, f'RMSE: {rmse_values[supplier]:.2f}', 
                 verticalalignment='center', size='small', color='black', weight='semibold')

    plt.title(f'Distribution of Errors for Each Supplier with RMSE Annotation (Suppliers {start_col + 1} to {end_col})')
    plt.xlabel('Error')
    plt.ylabel('Supplier ID')
    plt.tight_layout()

    # Save each figure with a unique filename
    save_path = os.path.join(save_dir, f'supplier_errors_with_rmse_{i + 1}.png')
    plt.savefig(save_path)
    plt.close()

print("Boxplots of supplier errors are saved")

# STEP 2.3
'''
To create a heatmap plot that shows the cost values as a matrix of tasks (rows) and suppliers (columns).
'''

# Create a heatmap of cost values
plt.figure(figsize=(20, 15))
sns.heatmap(costs_pivot, cmap='viridis', annot=False)

plt.title('Heatmap of Cost Values')
plt.xlabel('Supplier ID')
plt.ylabel('Task ID')
plt.tight_layout()

# Save the figure
save_path = os.path.join(save_dir, f'cost_values_heatmap.png')
plt.savefig(save_path)
plt.close()

print("Heatmap of cost values is saved")