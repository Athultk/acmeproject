# Machine Learning Model for Acme Coporation for selecting a supplier for their task

## Project Description
The project involves using three key datasets provided by Acme—task features, supplier characteristics, and a cost database—to build a machine learning model that can accurately predict and select the most cost-effective supplier. The model's success will be gauged by its error in predicting the cost of the chosen supplier compared to the actual lowest-cost supplier, with performance evaluated using root mean squared error (RMSE). The aim is to integrate machine learning concepts with practical business needs to improve cost estimation and supplier selection, thereby boosting operational efficiency and decision-making.

## Directory Structure


                
acmeproject/
    data/
        ── raw/
            ├── suppliers.csv
            ├── tasks.xlsx
            └── cost.csv
        ── 0_formatted/
            └── tasks_formatted.xlsx
        ── 1_filtered/
            └── tasks_filtered
        ── 2_reduced/
            ├── suppliers_reduced
            └── tasks_reduced
        ── 3_scaled/
            ├── tasks_scaled.xlsx
            └── suppliers_scaled
        ── 4_top_suppliers/
            └── Top_20_suppliers.csv
        ── 5_modeldata/
            ├── X.csv
            ├── y.csv
            └── groups.csv
        ── 6_merged/
            └── merged_data.csv
        
    reports/
        ── figures/
            ├── task_features_distribution_1.png
            ├── task_features_distribution_2.png
            ├── ...
            ├── supplier_errors_with_rmse_1.png
            ├── supplier_errors_with_rmse_2.png
            ├── ...
            └── cost_values_heatmap.png
        ── summary.txt
        
    models/
        ├── ridge_regression.pkl
        ├── random_forest.pkl
        ├── gradient_boosting.pkl
        └── models.py
    
    scripts/
        ├── 0_formatting/
        ├── 1_data_preparation/
        ├── 2_eda/
        ├── 3_mlmodel/
        ├── 4_crossvalidation/
        ├── 5_hyper/
        ├── 6_dashboard/
        └── dashboarddata.py
        
    src/
        ├── init.py
        ├── exception.py
        ├── logger.py
        └── utils.py
        
    venv/
    
    acmeproject.egg-info/
    
    .gitignore
    README.md
    requirements.txt
    setup.py

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Scripts_Description](#scripts_description)
    - [0_formatting](#0_formatting)
    - [1_data_preparation](#1_data_preparation)
    - [2_eda](#2_eda)
    - [3_mlmodel](#3_mlmodel)
    - [4_crossvalidation](#4_crossvalidation)
    - [5_hyper](#5_hyper)
    - [6_dashboard](#6_dashboard)
- [Authors_and_Acknowledgments](#Authors_and_Acknowledgments)
- [Contact_Information](#Contact_Information)

## Installation
1.1 Clone the repository:
git clone https://github.com/Athultk/acmeproject.git

1.2 Navigate to the project directory:
cd acmeproject

1.3 Install the required dependencies:
pip install -r requirements.txt

## Usage
To execute the scripts, run the following commands in your terminal:

1. Run the script to correct formatting:
python scripts/0_formatting.py

2. Run the script for data cleaning and processing:
python scripts/1_data_preparation.py

3. Run the script for exploratory data anlysis:
python scripts/2_eda.py

4. Run the script for ml model fitting and scoring:
python scripts/3_mlmodel.py

5. Run the script for cross validation:
python scripts/4_crossvalidation.py

6. Run the script for hyper-parameter optimization:
python scripts/5_hyper.py

7. Run the script for dashboard: (This is a draft code and yet to be fully materialised)
python scripts/dashboarddata.py
python scripts/dashboard.py

## Scripts_Description

### 0_formatting 

Script Overview: Correcting Task ID Format
File: 0_formatting.py

Description:
This script loads the tasks dataset, updates the 'Task ID' date format from 'YYYY MM DD' to 'DD/MM/YYYY' using the convert_date_format function, and saves the formatted dataset to a new Excel file.

Steps:
Load the tasks dataset.
Convert the 'Task ID' date format.
Save the formatted dataset to a new Excel file.

Output:
tasks_formatted.xlsx saved in the data/0_formatted directory.

### 1_data_preparation

Script Overview : Data Cleaning and Processing
File: 1_data_preparation.py

Description:
This script performs several data cleaning and processing tasks including verification of missing values, ID matching, feature statistics calculation, feature scaling, correlation analysis, and supplier filtering.

Steps:
Verify no missing values and ensure IDs match across datasets.
Remove tasks with missing cost values and save the filtered dataset.
Calculate statistics (max, min, mean, variance) for each feature and remove low variance columns.
Scale all features to the range [-1, 1].
Calculate and visualize the absolute correlation matrix and iteratively remove highly correlated features.
Identify the top 20 suppliers for each task and filter the suppliers dataset accordingly.

Outputs:
tasks_filtered.xlsx saved in the data/1_filtered directory.
tasks_reduced.xlsx and suppliers_reduced.csv saved in the data/2_reduced directory.
tasks_scaled.xlsx and suppliers_scaled.csv saved in the data/3_scaled directory.
Correlation matrices images saved in the reports/figures directory.
Top_20_suppliers.csv saved in the data/4_topsuppliers directory.

### 2_eda

Script Overview : Exploratory Data Analysis
File: 2_eda.py

Description:
Performs several data visualization tasks on provided datasets. It generates and saves visualizations to help understand the distribution of task features, supplier errors, and cost values. The script utilizes libraries like Pandas, Matplotlib, Seaborn, and NumPy.

Steps:
Creates multiple figures, each showing boxplots of task feature values. The features are split into chunks of 21 per figure, and each figure is saved as a PNG file in the reports/figures directory. This helps visualize the distribution of feature values across tasks.
Computes the errors for each supplier and calculates the Root Mean Square Error (RMSE) for each supplier. It then generates multiple figures showing boxplots of these errors, annotated with RMSE values. Each figure is saved as a PNG file in the reports/figures directory.
A heatmap is generated to visualize the cost values across tasks and suppliers. This heatmap is saved as a PNG file in the reports/figures directory, providing a matrix view of the cost distribution.

Outputs:
Boxplots of Task Features: Saved as task_features_distribution_1.png, task_features_distribution_2.png, etc., in the reports/figures directory.
Boxplots of Supplier Errors with RMSE: Saved as supplier_errors_with_rmse_1.png, supplier_errors_with_rmse_2.png, etc.
Heatmap of Cost Values: Saved as cost_values_heatmap.png.

### 3_mlmodel

Script Overview : ML Model fitting and scoring
File: 3_mlmodel.py

Description:
This script combines multiple datasets, trains various machine learning models, and evaluates their performance using Root Mean Square Error (RMSE). The models trained include Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor. It also saves the trained models for later use.

Steps:
Combines task features, supplier features, and costs into a single dataset. The combined dataset is cleaned, and relevant columns are selected and renamed for clarity. It then separates the data into feature variables (X), target variable (y), and task IDs. These datasets are saved as CSV files in the data/5_modeldata directory.
Splits the dataset into training and testing sets by randomly selecting 20 unique Task IDs for the test set. Masks are created to separate the data into training and testing sets for both features and target variables. The shapes of the resulting datasets are printed to verify the split.
Trains three machine learning models—Ridge Regression, Random Forest Regressor, and Gradient Boosting Regressor—on the training set. Each model is evaluated on the test set using RMSE. The trained models are saved to files using joblib.
Defines a function to calculate the error for each task based on the minimum cost and predicted costs. The script evaluates each model by loading it, making predictions on the test set, calculating errors, and computing the RMSE score for the predictions.

Outputs:
Prepared Data Files:
X.csv - Feature variables.
y.csv - Target variable (cost).
groups.csv - Task IDs.
These files are saved in the data/5_modeldata directory.

Model Files:
ridge_regression.pkl - Trained Ridge Regression model.
random_forest.pkl - Trained Random Forest Regressor model.
gradient_boosting.pkl - Trained Gradient Boosting Regressor model.
These files are saved in the models directory.

Evaluation Results:
Standard RMSE values for each model on the test set.
Errors and RMSE scores for each model, calculated based on task-specific errors.

### 4_crossvalidation

Script Overview : Cross validation
File: 3_crossvalidation.py

Description:

### 5_hyper

Script Overview : Hyper-parameter optimization
File: 5_hyper.py

Description:
This script performs hyperparameter tuning for two regression models: Ridge Regression and Random Forest Regressor. It uses custom scoring functions and Leave-One-Group-Out cross-validation to evaluate model performance and find the best hyperparameters.

Steps:
The script is divided into two main sections:

Ridge Regression Hyperparameter Tuning:
Loads feature variables (X), target variable (y), and task groups (groups).
Defines a custom scorer that calculates RMSE based on the error between predicted values and minimum actual costs for each group.
Iterates over a range of alpha values for Ridge Regression (0.01, 0.1, 1, 10, 100).
Uses Leave-One-Group-Out cross-validation to evaluate each alpha value.
Prints the best alpha and corresponding RMSE.

Random Forest Regressor Hyperparameter Tuning:
Uses the same custom scorer as Ridge Regression.
Defines a grid of hyperparameters for Random Forest Regressor, including n_estimators, max_depth, min_samples_split, and min_samples_leaf.
Performs grid search with Leave-One-Group-Out cross-validation using parallel processing to evaluate each combination of hyperparameters.
Finds and prints the best hyperparameters and their corresponding RMSE.
Retrains the Random Forest Regressor with the best hyperparameters on the entire dataset and saves the trained model to a file.

Outputs
Best Parameters and RMSE for Ridge Regression:
Printed to the console. This includes the best alpha value and its associated RMSE.
Best Parameters and RMSE for Random Forest Regressor:
Printed to the console. This includes the best hyperparameters and their associated RMSE.
Best Random Forest Model:
Saved as best_random_forest_model.pkl in the models directory.

### 6_dashboard

Script Overview : Dashboard (Draft one and not completed)
File: 6_dashboard.py

Description:
This Dash web application allows users to interactively predict and display the top 10 suppliers for a selected task using a Ridge Regression model. It leverages Dash components for user interaction and provides predictions based on the loaded model.

Steps: 
The script sets up a Dash web application to predict and display the top 10 suppliers for a selected task. The application uses a Ridge Regression model to make predictions based on the provided data.

Execution Steps
Load Data and Model:
The script loads the merged dataset and Ridge Regression model from specified file paths.
Prints file paths and sample data to ensure correctness.
Initialize Dash Application:
Defines the layout with a dropdown menu for task selection, a submit button, and a container to display results.
Define Callback:
Uses a callback function to update the output container based on the selected task and button click.
Filters the data for the selected task, makes predictions, and displays the top 10 suppliers based on the predicted cost.

Layout Components
Dropdown Menu (dcc.Dropdown):
Allows users to select a task from the list of available task IDs.
Submit Button (html.Button):
Triggers the prediction and display process when clicked.
Output Container (html.Div):
Displays a table of the top 10 suppliers based on predicted costs.

Callback Function
Inputs:
The number of button clicks and the selected task ID.
Processing:
Filters the data for the selected task.
Prepares feature columns for prediction.
Uses the Ridge Regression model to predict costs.
Identifies and displays the top 10 suppliers with the lowest predicted costs.

Running the Application
To run the application, execute the script. The Dash server will start, and you can access the dashboard by opening a web browser and navigating to http://127.0.0.1:8050/

## Authors_and_Acknowledgments
All scripts were equally contributed to by all three authors except for the ones mentioned below:

Athul Thattarkandy - Script: 3_mlmodel.py

Hongyi Yu - Script: 5_hyper.py;

Mert Ceylan - Script: 4_cross_validation.py

Special thanks to our Professor Yanfei Shan for the support and contributions.

## Contact_Information
For inquiries, please contact:

Athul Thattarkandy: athul.thattarkandy@tum.de

Hongyi Yu - hongyi.yu@tum.de

Mert Ceylan - mert.ceylan@tum.de
