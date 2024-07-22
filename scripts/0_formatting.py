import pandas as pd
import os

# STEP 1: CORRECTING THE TASK ID IN THE TASK FILE 
''' 
This script loads the tasks dataset, updates the 'Task ID' date format from 'YYYY MM DD' to 'DD/MM/YYYY' using 
the `convert_date_format` function, and saves the formatted dataset to a new Excel file.
'''

# Load the tasks dataset
scripts_dir = os.path.dirname(__file__)

# Construct the path to the tasks file
tasks_file_path = os.path.join(scripts_dir, '..', 'data', 'raw', 'tasks.xlsx')

# Read the tasks file
tasks_df = pd.read_excel(tasks_file_path)

# Print the first few rows of the dataframe
print(tasks_df.head())

# Function to convert date format from 'YYYY MM DD' to 'DD/MM/YYYY'
def convert_date_format(task_id):
    # Strip any leading/trailing whitespace
    task_id = task_id.strip()
    # Convert the date format
    date = pd.to_datetime(task_id, format='%Y %m %d')
    return date.strftime('%d/%m/%Y')

# Apply the function to the 'Task ID' column
tasks_df['Task ID'] = tasks_df['Task ID'].apply(convert_date_format)

# Construct the path to save the formatted tasks file
formatted_tasks_file_path = os.path.join(scripts_dir, '..', 'data', '0_formatted', 'tasks_formatted.xlsx')

# Save the updated tasks dataset back to an Excel file
tasks_df.to_excel(formatted_tasks_file_path, index=False)

# Read the formatted tasks file
tasks_formatted_df = pd.read_excel(formatted_tasks_file_path)

# Print the first few rows of the dataframe
print(tasks_formatted_df.head())

print("Task ID format updated and dataset saved.")