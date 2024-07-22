import pandas as pd
import numpy as np
import os

# Define the file paths for the datasets
scripts_dir = os.path.dirname(__file__)
tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'scaled', 'tasks_scaled.xlsx')
costs_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'cost.csv')
suppliers_file_path = os.path.join(scripts_dir, '..', 'data', 'topsuppliers', 'Top_20_suppliers.csv')

# Load the datasets
tasks_df = pd.read_excel(tasks_file_path)
costs_df = pd.read_csv(costs_file_path)
suppliers_df = pd.read_csv(suppliers_file_path)

# Merge tasks with costs
merged_df = pd.merge(costs_df, tasks_df, on='Task ID')

# Merge with suppliers
final_df = pd.merge(merged_df, suppliers_df, left_on='Supplier ID', right_on='Supplier ID')

# Select relevant columns
final_df = final_df[['Task ID'] + list(tasks_df.columns[1:]) + list(suppliers_df.columns[1:]) + ['Cost']]

# Rename columns for clarity
final_df.columns = final_df.columns.str.replace(' ', '_')

# Save the merged data to a CSV file
merged_data_path = os.path.join(scripts_dir, '..', 'data', '6_merged', 'merged_data.csv')
final_df.to_csv(merged_data_path, index=False)

print(f"Prepared data saved to merged folder")
