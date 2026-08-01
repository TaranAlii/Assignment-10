import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("--- Task 1: Data Understanding and Preprocessing ---")
    
    # 1. Load the dataset
    data_path = "heart.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("assignment-10", "heart.csv")
    
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # 2. Display the first five records
    print("\nFirst 5 records of the dataset:")
    print(df.head())
    
    # 3. Identify Numerical Features and Target Variable
    # All features except target are numerical clinical parameters in this dataset.
    features = list(df.columns[:-1])
    target = df.columns[-1]
    print(f"\nFeatures (Numerical Clinical Parameters):")
    print(features)
    print(f"Target Variable: {target}")
    
    # 4. Check for missing values
    print("\nChecking for missing values in each column:")
    missing_values = df.isnull().sum()
    print(missing_values)
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing}")
    
    # 5. Split dataset into 80% training and 20% testing
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nSplitting dataset (80% training, 20% testing):")
    print(f"  - Training set size: {X_train.shape[0]} samples")
    print(f"  - Testing set size: {X_test.shape[0]} samples")
    
    print("\n--- Task 2: Model Development ---")
    # Build a Random Forest classification model
    print("Initializing RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    
    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model accuracy...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy Score: {accuracy * 100:.2f}%")
    
    # Save the trained model
    model_filename = "model.pkl"
    # Make sure we save it in the current directory or target folder
    print(f"Saving the trained model to: {model_filename}")
    joblib.dump(model, model_filename)
    
    print("\nModel training and serialization completed successfully!")

if __name__ == "__main__":
    main()
