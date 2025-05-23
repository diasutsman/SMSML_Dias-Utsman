#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Automates data preprocessing for the Iris dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """
    Load Iris dataset and save raw version
    """
    print("Loading Iris dataset...")
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    # Create and save raw dataset
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "namadataset_raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    iris_df = pd.concat([X, y], axis=1)
    raw_path = os.path.join(raw_dir, "iris_raw.csv")
    iris_df.to_csv(raw_path, index=False)
    
    print(f"Raw data saved to: {raw_path}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y, iris_df

def check_missing_values(X):
    """
    Check for missing values in the dataset
    """
    print("\nChecking for missing values...")
    missing_values = X.isnull().sum()
    print(f"Missing values per feature: {missing_values.to_dict()}")
    return missing_values

def detect_outliers(X):
    """
    Detect outliers using the IQR method
    """
    print("\nDetecting outliers using IQR method...")
    outliers = {}
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = X[(X[col] < lower_bound) | (X[col] > upper_bound)].shape[0]
    
    for col, count in outliers.items():
        print(f"{col}: {count} outliers")
    
    return outliers

def standardize_features(X):
    """
    Standardize features using StandardScaler
    """
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Statistics after standardization:")
    print(X_scaled_df.describe().loc[['mean', 'std'], :].round(2))
    
    return X_scaled, scaler

def prepare_train_test_split(X_scaled, y):
    """
    Split data into training and testing sets
    """
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_scaled, y):
    """
    Save preprocessed data to disk
    """
    # Create DataFrame for preprocessed data
    X_scaled_df = pd.DataFrame(X_scaled, columns=load_iris().feature_names)
    preprocessed_df = pd.concat([X_scaled_df, y], axis=1)
    
    # Create directory for preprocessed data
    output_dir = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed data
    output_path = os.path.join(output_dir, "iris_preprocessed.csv")
    preprocessed_df.to_csv(output_path, index=False)
    
    print(f"\nPreprocessed data saved to: {output_path}")
    return output_path

def generate_visualization(X, y, save_path=None):
    """
    Generate visualizations for data exploration
    """
    print("\nGenerating visualization...")
    
    # Create a directory for visualizations
    vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Combine X and y for easier plotting
    iris_df = pd.concat([X, y], axis=1)
    
    # Pairplot for feature relationships
    plt.figure(figsize=(10, 8))
    sns.pairplot(iris_df, hue='species')
    plt.suptitle('Feature Relationships', y=1.02)
    pairplot_path = os.path.join(vis_dir, "pairplot.png")
    plt.savefig(pairplot_path)
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    corr_path = os.path.join(vis_dir, "correlation.png")
    plt.savefig(corr_path)
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")
    return vis_dir

def main():
    """
    Main function to run the data preprocessing pipeline
    """
    print("Starting data preprocessing pipeline...\n")
    
    # Step 1: Load data
    X, y, iris_df = load_data()
    
    # Step 2: Check for missing values
    missing_values = check_missing_values(X)
    
    # Step 3: Detect outliers
    outliers = detect_outliers(X)
    
    # Step 4: Standardize features
    X_scaled, scaler = standardize_features(X)
    
    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = prepare_train_test_split(X_scaled, y)
    
    # Step 6: Save preprocessed data
    output_path = save_preprocessed_data(X_scaled, y)
    
    # Step 7: Generate visualizations
    vis_dir = generate_visualization(X, y)
    
    print("\nData preprocessing completed successfully!")
    print(f"Preprocessed data saved to: {output_path}")
    print(f"Visualizations saved to: {vis_dir}")
    
    # Return data for potential future use
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'output_path': output_path
    }

if __name__ == "__main__":
    main()
