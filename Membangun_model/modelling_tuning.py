#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Advanced Model Training with Hyperparameter Tuning and Manual MLflow Logging
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

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

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def tune_hyperparameters(X_train, y_train):
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
    rf = RandomForestClassifier(random_state=42)
    
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
    
    return grid_search, best_params, best_score

def train_model_with_best_params(best_params, X_train, y_train):
    """
    Train model with best parameters
    """
    print("\nTraining model with best parameters...")
    
    model = RandomForestClassifier(
        **best_params,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance with comprehensive metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # For ROC curve and AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None
    
    # Calculate basic metrics
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
    
    # Additional metrics for more comprehensive evaluation
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Calculate class-specific metrics
    classes = np.unique(y_test)
    class_metrics = {}
    
    for i, cls in enumerate(classes):
        cls_accuracy = accuracy_score(y_test == cls, y_pred == cls)
        cls_precision = precision_score(y_test == cls, y_pred == cls, zero_division=0)
        cls_recall = recall_score(y_test == cls, y_pred == cls, zero_division=0)
        cls_f1 = f1_score(y_test == cls, y_pred == cls, zero_division=0)
        
        class_metrics[f'class_{cls}'] = {
            'accuracy': cls_accuracy,
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1
        }
    
    metrics['class_metrics'] = class_metrics
    
    # Calculate ROC and AUC for each class (one-vs-rest)
    if y_prob is not None:
        roc_auc = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test == cls, y_prob[:, i])
            roc_auc[f'class_{cls}'] = auc(fpr, tpr)
        
        metrics['roc_auc'] = roc_auc
        metrics['avg_roc_auc'] = np.mean(list(roc_auc.values()))
    
    return metrics

def log_metrics_and_artifacts(metrics, model, X, feature_names, run_id):
    """
    Log comprehensive metrics and artifacts to MLflow
    """
    # Create artifacts directory
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Log basic metrics
    mlflow.log_metric("accuracy", metrics['accuracy'])
    mlflow.log_metric("precision", metrics['precision'])
    mlflow.log_metric("recall", metrics['recall'])
    mlflow.log_metric("f1", metrics['f1'])
    
    # Log class-specific metrics
    for cls, cls_metrics in metrics['class_metrics'].items():
        for metric_name, value in cls_metrics.items():
            mlflow.log_metric(f"{cls}_{metric_name}", value)
    
    # Log ROC AUC metrics if available
    if 'roc_auc' in metrics:
        for cls, auc_value in metrics['roc_auc'].items():
            mlflow.log_metric(f"{cls}_roc_auc", auc_value)
        mlflow.log_metric("avg_roc_auc", metrics['avg_roc_auc'])
    
    # Plot and log confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f"{artifacts_dir}/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    
    # Plot and log feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importances')
    plt.tight_layout()
    fi_path = f"{artifacts_dir}/feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)
    
    # Create and log feature importance CSV
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    importance_path = f"{artifacts_dir}/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path)
    
    # Plot and log ROC curves if probabilities are available
    if 'roc_auc' in metrics:
        plt.figure(figsize=(10, 8))
        for i, cls in enumerate(metrics['roc_auc'].keys()):
            if cls.startswith('class_'):
                class_idx = int(cls.split('_')[1])
                fpr, tpr, _ = roc_curve(
                    (metrics['y_test'] == class_idx).astype(int), 
                    metrics['y_prob'][:, class_idx]
                )
                plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {metrics["roc_auc"][cls]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        roc_path = f"{artifacts_dir}/roc_curves.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
    
    # Save and log model
    model_path = f"{artifacts_dir}/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    
    # Log model with MLflow
    mlflow.sklearn.log_model(model, "model")
    
    # Log hyperparameters
    mlflow.log_params(model.get_params())
    
    # Log run ID
    mlflow.log_param("run_id", run_id)
    
    # Log run timestamp
    mlflow.log_param("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Log dataset information
    mlflow.log_param("dataset_shape", X.shape)
    mlflow.log_param("n_features", X.shape[1])

def main():
    """
    Main function to run the model training pipeline with hyperparameter tuning
    """
    # Create experiment
    experiment_name = "Iris-Classification-Tuned"
    mlflow.set_experiment(experiment_name)
    
    # Define relative path to the preprocessed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "namadataset_preprocessing", "iris_preprocessed.csv")
    
    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest-Tuned") as run:
        run_id = run.info.run_id
        
        # Load data
        X, y = load_preprocessed_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Tune hyperparameters
        grid_search, best_params, best_score = tune_hyperparameters(X_train, y_train)
        
        # Log best parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log best cross-validation score
        mlflow.log_metric("best_cv_score", best_score)
        
        # Train model with best parameters
        model = train_model_with_best_params(best_params, X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Add test data to metrics for ROC curve plotting
        metrics['y_test'] = y_test
        
        # Log metrics and artifacts
        log_metrics_and_artifacts(metrics, model, X, X.columns, run_id)
        
        print(f"\nModel training completed successfully!")
        print(f"Model artifacts logged to MLflow")
        print(f"Run ID: {run_id}")
        
        # Return important information
        return {
            'run_id': run_id,
            'best_params': best_params,
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
        }

if __name__ == "__main__":
    main()
