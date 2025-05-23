#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Train a simple model for Iris classification
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("./mlruns")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model to a file
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Log the model with MLflow
with mlflow.start_run(run_name="iris_model"):
    # Log model parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Log model performance
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    # Log sample data as artifacts
    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=iris.feature_names + ['target']
    )
    iris_df.to_csv("iris_data.csv", index=False)
    mlflow.log_artifact("iris_data.csv")
    
    # Print run info
    print(f"Model training completed with accuracy: {accuracy:.4f}")
    print(f"Model saved to: model/model.pkl")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Experiment ID: {mlflow.active_run().info.experiment_id}")

print("Model training and logging completed successfully!")
