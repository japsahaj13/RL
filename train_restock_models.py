#!/usr/bin/env python3
"""
Train Restock Models with Improved Data

This script trains restock parameter prediction models using the improved retail data
with explicit inventory management rules.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from utilities.restock_prediction import RestockPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='restock_model_training.log',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Also output to console

def evaluate_model(y_true, y_pred, feature_names=None, feature_importances=None):
    """
    Evaluate model performance and log results.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        feature_names: Names of features
        feature_importances: Feature importances from model
    
    Returns:
        Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Model metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    if feature_names is not None and feature_importances is not None:
        # Log top feature importances
        importances = list(zip(feature_names, feature_importances))
        importances.sort(key=lambda x: x[1], reverse=True)
        top_features = importances[:3]
        logger.info(f"Top 3 important features: {top_features}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def train_restock_models():
    """
    Train restock parameter prediction models.
    
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    logger.info("Starting restock prediction model training")
    
    # Initialize RestockPredictor
    predictor = RestockPredictor(data_path='data/improved_retail_store_inventory.csv')
    
    # Load and preprocess data
    df = predictor.load_data()
    logger.info(f"Loaded {len(df)} records with categories: {', '.join(df['Category'].unique())}")
    
    # Prepare training data
    training_data = predictor.prepare_training_data(df)
    
    # Define feature names for interpretability
    feature_names = [
        'is_electronics', 'is_furniture', 'is_clothing', 'is_toys', 'is_groceries',
        'demand_forecast', 'inventory_level', 'price', 'units_sold',
        'actual_demand', 'lead_time_days', 'safety_stock',
        'day_of_week', 'month'
    ]
    
    # Train models with evaluation
    logger.info("Training restock parameter prediction models")
    
    models = {}
    metrics = {}
    
    for param_name, (X, y) in training_data.items():
        logger.info(f"Training model for {param_name} prediction")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Evaluate model
        metrics[param_name] = evaluate_model(
            y_test, y_pred, 
            feature_names=feature_names,
            feature_importances=model.feature_importances_
        )
        
        # Save model
        models[param_name] = model
        model_path = f"models/saved/restock_{param_name}_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model for {param_name} saved to {model_path}")
    
    # Save all models and metrics in one package
    model_data = {
        'models': models,
        'metrics': metrics,
        'feature_names': feature_names
    }
    
    with open('models/saved/restock_models_package.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("Restock prediction model training complete")
    
    return model_data

if __name__ == "__main__":
    train_restock_models() 