#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Model Training Script with MLflow Project Structure
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model for Iris classification")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="namadataset_preprocessing/iris_preprocessed.csv",
        help="Path to preprocessed data"
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--tune_hyperparameters", 
        type=bool, 
        default=True,
        help="Whether to perform hyperparameter tuning"
    )
    return parser.parse_args()

def load_preprocessed_data(data_path):
    """
    Load preprocessed data from CSV file
    """
    print(f"Loading preprocessed data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # Split features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_basic_model(X_train, y_train, random_state=42):
    """
    Train a basic Random Forest classifier
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(X_train, y_train, random_state=42):
    """
    Tune hyperparameters using GridSearchCV
    """
    print("\nTuning hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=random_state)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return grid_search.best_estimator_, best_params, best_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Return metrics and predictions
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, classes=None):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/confusion_matrix.png')
    plt.close()

def log_feature_importance(model, feature_names):
    """
    Log feature importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importances')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/feature_importance.png')
    plt.close()
    
    # Create a DataFrame for easier logging
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    return importance_df

def main():
    """
    Main function to run the model training pipeline
    """
    # Parse command line arguments
    args = parse_args()
    
    # Create experiment
    experiment_name = "Iris-Classification-CI"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    run_name = "RandomForest-Tuned" if args.tune_hyperparameters else "RandomForest-Basic"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Load data
        X, y = load_preprocessed_data(args.data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)
        
        # Log parameters
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("tune_hyperparameters", args.tune_hyperparameters)
        
        # Train model
        if args.tune_hyperparameters:
            print("\nPerforming hyperparameter tuning...")
            model, best_params, best_score = tune_hyperparameters(X_train, y_train, args.random_state)
            
            # Log best parameters and score
            for param, value in best_params.items():
                mlflow.log_param(param, value)
            mlflow.log_metric("best_cv_score", best_score)
        else:
            print("\nTraining basic model without hyperparameter tuning...")
            model = train_basic_model(X_train, y_train, args.random_state)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1", metrics['f1'])
        
        # Track model training time as a custom metric
        mlflow.log_metric("training_samples", X_train.shape[0])
        mlflow.log_metric("testing_samples", X_test.shape[0])
        
        # Plot and log confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        
        # Log feature importance
        importance_df = log_feature_importance(model, X.columns)
        importance_path = "artifacts/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact("artifacts/feature_importance.png")
        mlflow.log_artifact(importance_path)
        
        # Save model
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Register model with MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nModel training completed successfully!")
        print(f"Model artifacts logged to MLflow")
        print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
