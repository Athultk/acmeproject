import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import os

'''
Please note that this was our attempt to create a dashboard. However the same is not working properly. 
Please run dashboarddata.py before this.
Sets up a Dash web application to interactively predict and display the top 10 suppliers for a selected task based 
on a Ridge Regression model. Loads the data and model, and defines a layout with a dropdown and a button. 
Uses a callback to filter data, make predictions, and display the top suppliers when the button is clicked.
'''

# Load data and model
scripts_dir = os.path.dirname(__file__)
merged_data_path = os.path.join(scripts_dir, '..', 'data', '6_merged', 'merged_data.csv')
model_path = os.path.join(scripts_dir, '..', 'models', 'ridge_regression.pkl')

# Print file paths to ensure they are correct
print(f"Merged data path: {merged_data_path}")
print(f"Model path: {model_path}")

# Load the prepared data
merged_data = pd.read_csv(merged_data_path)

# Print the first few rows of the data to ensure it loads correctly
print("Merged data sample:")
print(merged_data.head())

# Load the Ridge Regression model
model = joblib.load(model_path)

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    dcc.Dropdown(
        id='task-dropdown',
        options=[{'label': str(task), 'value': task} for task in merged_data['Task_ID'].unique()],
        placeholder="Select a task"
    ),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-container')
])

# Callback
@app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('task-dropdown', 'value')]
)
def update_output(n_clicks, task_id):
    print(f"Button clicked {n_clicks} times.")
    print(f"Selected Task ID: {task_id}")
    
    if n_clicks > 0 and task_id:
        # Filter data for the selected task
        task_data = merged_data[merged_data['Task_ID'] == task_id]
        print(f"Filtered task data:")
        print(task_data.head())
        
        X_task = task_data.drop(columns=['Task_ID', 'Supplier_ID', 'Cost'])
        
        # Ensure that the feature columns are correctly prepared
        print("Feature columns for prediction:")
        print(X_task.head())
        
        # Predict costs
        task_data['Predicted_Cost'] = model.predict(X_task)
        
        # Get top 10 suppliers
        top_suppliers = task_data.nsmallest(10, 'Predicted_Cost')
        print("Top 10 suppliers based on predicted cost:")
        print(top_suppliers.head())
        
        return html.Table([
            html.Tr([html.Th(col) for col in top_suppliers.columns])] +
            [html.Tr([html.Td(top_suppliers.iloc[i][col]) for col in top_suppliers.columns])
             for i in range(min(len(top_suppliers), 10))])
    return "Select a task and click Submit"

if __name__ == '__main__':
    app.run_server(debug=True)
