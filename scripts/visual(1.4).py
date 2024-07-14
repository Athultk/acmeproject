import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

# Load the datasets
scripts_dir = os.path.dirname(__file__)

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
