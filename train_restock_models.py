#!/usr/bin/env python3
"""
Restock Model Training Script

This script trains the restock parameter prediction models (level, amount, period)
using retail inventory data and saves them to the models directory.

Run this script after cloning the repository to create the necessary model files:
python train_restock_models.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('restock_model_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = 'data/retail_store_inventory.csv'
MODEL_DIR = 'models'
FEATURE_IMPORTANCE_DIR = 'data_analysis/figures'

def check_data_exists() -> bool:
    """Check if the retail inventory data file exists."""
    if not os.path.exists(DATA_PATH):
        logger.error(f"Retail data file not found: {DATA_PATH}")
        logger.error(f"Please make sure the data file exists at {DATA_PATH}")
        return False
    return True

def load_data() -> Optional[pd.DataFrame]:
    """
    Load and preprocess the retail inventory dataset.
    
    Returns:
        Preprocessed DataFrame or None if data file not found
    """
    if not check_data_exists():
        return None
        
    logger.info(f"Loading retail data from {DATA_PATH}")
    
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter out rows with missing key values
        key_columns = ['Category', 'Inventory Level', 'Units Sold', 'Units Ordered', 
                       'Demand Forecast', 'Price']
        df = df.dropna(subset=key_columns)
        
        # Sort by category and date
        df = df.sort_values(['Category', 'Store ID', 'Product ID', 'Date'])
        
        logger.info(f"Loaded {len(df):,} records with categories: {', '.join(df['Category'].unique())}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def prepare_training_data(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare training data for restock parameter prediction models.
    
    Args:
        df: Preprocessed retail DataFrame
        
    Returns:
        Dictionary with training data for each parameter model
    """
    logger.info("Preparing training data for restock parameter models")
    
    # Extract features from dataframe
    features = []
    feature_names = []
    targets = {
        'level': [],
        'amount': [],
        'period': []
    }
    
    # Group by category, store, and product
    group_count = 0
    restock_event_count = 0
    
    for (category, store_id, product_id), group in df.groupby(['Category', 'Store ID', 'Product ID']):
        group_count += 1
        
        # Sort by date
        group = group.sort_values('Date')
        
        # Find restock events (when Units Ordered > 0)
        restock_events = group[group['Units Ordered'] > 0].copy()
        
        if len(restock_events) < 2:
            continue
            
        restock_event_count += len(restock_events) - 1
        
        # Calculate restock periods
        restock_events['Next Restock Date'] = restock_events['Date'].shift(-1)
        restock_events['Days Between Restocks'] = (restock_events['Next Restock Date'] - 
                                                  restock_events['Date']).dt.days
        
        # Calculate inventory levels at time of restock
        restock_events['Inventory Before Restock'] = (restock_events['Inventory Level'] - 
                                                     restock_events['Units Ordered'])
        
        # Define feature names if not already done
        if not feature_names:
            feature_names = [
                'is_Electronics',
                'is_Furniture',
                'is_Clothing',
                'is_Toys',
                'is_Groceries',
                'demand_forecast',
                'inventory_level',
                'price',
                'units_sold',
                'day_of_week',
                'month'
            ]
        
        # For each restock event except the last one (which doesn't have next restock date)
        for i in range(len(restock_events) - 1):
            event = restock_events.iloc[i]
            
            # Extract features
            feature_vector = [
                1 if category == "Electronics" else 0,
                1 if category == "Furniture" else 0,
                1 if category == "Clothing" else 0,
                1 if category == "Toys" else 0,
                1 if category == "Groceries" else 0,
                event['Demand Forecast'],
                event['Inventory Level'],
                event['Price'],
                event['Units Sold'],
                event['Date'].dayofweek,
                event['Date'].month
            ]
            
            features.append(feature_vector)
            targets['level'].append(event['Inventory Before Restock'])
            targets['amount'].append(event['Units Ordered'])
            targets['period'].append(event['Days Between Restocks'])
    
    # Convert to numpy arrays
    X = np.array(features)
    y_level = np.array(targets['level'])
    y_amount = np.array(targets['amount'])
    y_period = np.array(targets['period'])
    
    logger.info(f"Processed {group_count} product groups with {restock_event_count} restock events")
    logger.info(f"Prepared {len(features)} training samples")
    
    return {
        'level': (X, y_level, feature_names),
        'amount': (X, y_amount, feature_names),
        'period': (X, y_period, feature_names)
    }

def train_and_save_models(training_data: Dict[str, Tuple[np.ndarray, np.ndarray, list]]) -> Dict[str, Dict]:
    """
    Train prediction models for restock parameters and save them to disk.
    
    Args:
        training_data: Dictionary with training data for each parameter model
        
    Returns:
        Dictionary with model information
    """
    logger.info("Training restock parameter prediction models")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
    
    model_info = {}
    
    for param_name, (X, y, feature_names) in training_data.items():
        logger.info(f"Training model for {param_name} prediction")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline with preprocessing and model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importances
        importances = model.named_steps['model'].feature_importances_
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'restock_{param_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Store model information
        model_info[param_name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'feature_importances': dict(zip(feature_names, importances)),
            'path': model_path
        }
        
        logger.info(f"Model for {param_name} saved to {model_path}")
        logger.info(f"Model metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        logger.info(f"Top 3 important features: {sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]}")
    
    return model_info

def print_summary(model_info: Dict[str, Dict]) -> None:
    """Print a summary of the trained models."""
    print("\n" + "="*80)
    print("RESTOCK PREDICTION MODEL TRAINING SUMMARY")
    print("="*80)
    
    for param_name, info in model_info.items():
        print(f"\nModel: restock_{param_name}_model.pkl")
        print(f"  RMSE: {info['rmse']:.2f}")
        print(f"  R²: {info['r2']:.2f}")
        print(f"  Training time: {info['training_time']:.2f} seconds")
        
        print("  Top 5 feature importances:")
        sorted_importances = sorted(
            info['feature_importances'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for feat, imp in sorted_importances[:5]:
            print(f"    - {feat}: {imp:.4f}")
    
    print("\nModels saved to:", MODEL_DIR)
    print("="*80)

def main():
    """Main execution function."""
    logger.info("Starting restock prediction model training")
    
    # Load and preprocess data
    df = load_data()
    if df is None:
        return
    
    # Prepare training data
    training_data = prepare_training_data(df)
    
    # Train and save models
    model_info = train_and_save_models(training_data)
    
    # Print summary
    print_summary(model_info)
    
    logger.info("Restock prediction model training complete")

if __name__ == "__main__":
    main() 