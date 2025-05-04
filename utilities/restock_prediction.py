"""
Restock Parameter Prediction Module

This module provides functions to predict restock parameters from retail data,
ensuring predictions are only made after the previous restock period is exhausted.
The model is pre-trained on historical data and used during simulation.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache for trained models
_model_cache = {}

class RestockPredictor:
    """
    Predicts optimal restock parameters based on historical data.
    Makes new predictions only after the previous restock period is exhausted.
    """
    
    def __init__(self, data_path: str = 'data/improved_retail_store_inventory.csv'):
        """
        Initialize the restock predictor with data.
        
        Args:
            data_path: Path to the retail inventory data
        """
        self.last_restock_time = {}  # Tracks last restock time by category
        self.current_parameters = {}  # Current parameters by category
        self.data_path = data_path
        self.models = {
            'level': None,    # Model to predict restock level
            'amount': None,   # Model to predict restock amount
            'period': None    # Model to predict restock period
        }
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the retail inventory dataset.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading retail data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Retail data file not found: {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter out rows with missing key values
        key_columns = ['Category', 'Inventory Level', 'Units Sold', 'Units Ordered', 
                       'Demand Forecast', 'Price']
        df = df.dropna(subset=key_columns)
        
        # Sort by category and date
        df = df.sort_values(['Category', 'Store ID', 'Product ID', 'Date'])
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data for restock parameter prediction models using improved dataset.
        
        Args:
            df: Preprocessed retail DataFrame
            
        Returns:
            Dictionary with training data for each parameter model
        """
        logger.info("Preparing training data for restock parameter models")
        
        # Extract features from dataframe
        features = []
        targets = {
            'level': [],
            'amount': [],
            'period': []
        }
        
        # Group by category, store, and product
        for (category, store_id, product_id), group in df.groupby(['Category', 'Store ID', 'Product ID']):
            # Sort by date
            group = group.sort_values('Date')
            
            # Find restock events (when Units Ordered > 0)
            restock_events = group[group['Units Ordered'] > 0].copy()
            
            if len(restock_events) < 2:
                continue
            
            # For each restock event
            for i in range(len(restock_events)):
                event = restock_events.iloc[i]
                
                # Extract features
                feature_vector = [
                    # Categoricals would be one-hot encoded in a full implementation
                    1 if category == "Electronics" else 0,
                    1 if category == "Furniture" else 0,
                    1 if category == "Clothing" else 0,
                    1 if category == "Toys" else 0,
                    1 if category == "Groceries" else 0,
                    event['Demand Forecast'],
                    event['Inventory Level'],
                    event['Price'],
                    event['Units Sold'],
                    # Use actual columns from the improved dataset
                    event['Actual Demand'],
                    event['Lead Time Days'],
                    event['Safety Stock'],
                    # Day of week, month, etc.
                    pd.to_datetime(event['Date']).dayofweek,
                    pd.to_datetime(event['Date']).month
                ]
                
                features.append(feature_vector)
                
                # Use explicit reorder point for level target
                targets['level'].append(event['Reorder Point'])
                
                # Use EOQ for amount target
                targets['amount'].append(event['EOQ'])
                
                # Use lead time for period target (frequency of checking inventory)
                targets['period'].append(max(1, event['Lead Time Days'] * 2))  # Check at least twice as often as lead time
        
        # Convert to numpy arrays
        X = np.array(features)
        y_level = np.array(targets['level'])
        y_amount = np.array(targets['amount'])
        y_period = np.array(targets['period'])
        
        logger.info(f"Prepared {len(features)} training samples")
        
        return {
            'level': (X, y_level),
            'amount': (X, y_amount),
            'period': (X, y_period)
        }
    
    def train_models(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Train prediction models for restock parameters.
        
        Args:
            training_data: Dictionary with training data for each parameter model
        """
        logger.info("Training restock parameter prediction models")
        
        for param_name, (X, y) in training_data.items():
            logger.info(f"Training model for {param_name} prediction")
            
            # Create pipeline with preprocessing and model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Train model
            pipeline.fit(X, y)
            
            # Store model
            self.models[param_name] = pipeline
            
            logger.info(f"Model for {param_name} trained successfully")
    
    def save_models(self, model_dir: str = 'models') -> None:
        """
        Save trained models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        for param_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(model_dir, f'restock_{param_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model for {param_name} saved to {model_path}")
    
    def load_models(self, model_dir: str = 'models/saved') -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory where models are saved
        """
        print(f"Looking for restock prediction models in {os.path.abspath(model_dir)}")
        
        # Check both models/saved and models directories
        model_dirs = [model_dir, 'models']
        
        models_loaded = False
        for dir_path in model_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for param_name in ['level', 'amount', 'period']:
                model_path = os.path.join(dir_path, f'restock_{param_name}_model.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[param_name] = pickle.load(f)
                    print(f"Loaded restock {param_name} model from {os.path.abspath(model_path)}")
                    models_loaded = True
            
            if models_loaded:
                break
        
        if not models_loaded:
            print("No restock prediction models found. Will use default parameters.")
    
    def train_and_save(self) -> None:
        """
        Train all models and save them to disk.
        """
        df = self.load_data()
        training_data = self.prepare_training_data(df)
        self.train_models(training_data)
        self.save_models()
    
    def prepare_feature_vector(self, 
                              category: str,
                              demand_forecast: float,
                              inventory_level: float,
                              price: float,
                              units_sold: float,
                              current_date=None) -> np.ndarray:
        """
        Prepare feature vector for prediction using improved feature set.
        
        Args:
            category: Product category
            demand_forecast: Current demand forecast
            inventory_level: Current inventory level
            price: Current price
            units_sold: Recent units sold
            current_date: Current date (defaults to today)
            
        Returns:
            Feature vector for model input
        """
        if current_date is None:
            current_date = pd.Timestamp.now()
        
        # Estimate other parameters that would be in the improved dataset
        actual_demand = demand_forecast * (0.9 + 0.2 * random.random())  # Approx
        lead_time_days = 2 if category == "Groceries" else 3  # Reasonable defaults
        if category == "Furniture":
            lead_time_days = 5
        
        # Estimate safety stock based on demand variability
        safety_stock = demand_forecast * 0.2  # Assuming 20% of forecast as safety stock
        
        # Create feature vector matching the improved data structure
        feature_vector = [
            1 if category == "Electronics" else 0,
            1 if category == "Furniture" else 0,
            1 if category == "Clothing" else 0,
            1 if category == "Toys" else 0,
            1 if category == "Groceries" else 0,
            demand_forecast,
            inventory_level,
            price,
            units_sold,
            actual_demand,
            lead_time_days,
            safety_stock,
            current_date.dayofweek,
            current_date.month
        ]
        
        return np.array([feature_vector])
    
    def predict_restock_parameters(self,
                                 category: str,
                                 demand_forecast: float,
                                 inventory_level: float,
                                 price: float,
                                 units_sold: float,
                                 current_time: int,
                                 current_date=None) -> Optional[Dict[str, int]]:
        """
        Predict restock parameters if the previous restock period is exhausted.
        
        Args:
            category: Product category
            demand_forecast: Current demand forecast
            inventory_level: Current inventory level
            price: Current price
            units_sold: Recent units sold
            current_time: Current time step in the simulation
            current_date: Current date (defaults to today)
            
        Returns:
            Dictionary with predicted restock parameters or None if not time to restock
        """
        # Check if we need to initialize parameters for this category
        if category not in self.current_parameters:
            # Initialize with predictions
            X = self.prepare_feature_vector(category, demand_forecast, inventory_level, 
                                           price, units_sold, current_date)
            
            # Set initial parameters
            init_period = max(1, int(self.models['period'].predict(X)[0]))
            init_level = max(5, int(self.models['level'].predict(X)[0]))
            init_amount = max(10, int(self.models['amount'].predict(X)[0]))
            
            self.current_parameters[category] = {
                'restock_period': init_period,
                'restock_level': init_level,
                'restock_amount': init_amount
            }
            self.last_restock_time[category] = current_time
            
            return self.current_parameters[category]
        
        # Check if the previous restock period has been exhausted
        last_restock = self.last_restock_time.get(category, 0)
        current_period = self.current_parameters[category]['restock_period']
        
        # Only predict new parameters if the previous period is exhausted
        if current_time - last_restock >= current_period:
            # Create feature vector
            X = self.prepare_feature_vector(category, demand_forecast, inventory_level, 
                                           price, units_sold, current_date)
            
            # Make predictions
            level = max(5, int(self.models['level'].predict(X)[0]))
            amount = max(10, int(self.models['amount'].predict(X)[0]))
            period = max(1, int(self.models['period'].predict(X)[0]))
            
            # Update parameters
            self.current_parameters[category] = {
                'restock_period': period,
                'restock_level': level,
                'restock_amount': amount
            }
            self.last_restock_time[category] = current_time
            
            logger.info(f"New restock parameters for {category}: level={level}, amount={amount}, period={period}")
            return self.current_parameters[category]
        
        # If not time to predict new parameters, return None
        return None
    
    def get_current_parameters(self, category: str) -> Dict[str, int]:
        """
        Get the current restock parameters for a category.
        
        Args:
            category: Product category
            
        Returns:
            Dictionary with current restock parameters
        """
        if category not in self.current_parameters:
            # Default values if no predictions made yet
            return {
                'restock_period': 6,
                'restock_level': 30,
                'restock_amount': 100
            }
        
        return self.current_parameters[category]

# Global instance for access from config
_restock_predictor = None

def get_restock_predictor() -> RestockPredictor:
    """
    Get the global restock predictor instance.
    
    Returns:
        RestockPredictor instance
    """
    global _restock_predictor
    
    if _restock_predictor is None:
        print("Initializing restock prediction system...")
        _restock_predictor = RestockPredictor()
        
        # Try to load pre-trained models
        try:
            _restock_predictor.load_models()
            print("Successfully loaded restock prediction models")
        except Exception as e:
            print(f"Failed to load pre-trained restock models: {e}")
            print("Will use default restock parameters")
    
    return _restock_predictor

def predict_restock_parameters(
    category: str,
    demand_forecast: float,
    inventory_level: float,
    price: float,
    units_sold: float,
    current_time: int,
    current_date=None
) -> Optional[Dict[str, int]]:
    """
    Predict restock parameters if the previous restock period is exhausted.
    
    Args:
        category: Product category
        demand_forecast: Current demand forecast
        inventory_level: Current inventory level
        price: Current price
        units_sold: Recent units sold
        current_time: Current time step in the simulation
        current_date: Current date (defaults to today)
        
    Returns:
        Dictionary with predicted restock parameters or None if not time to restock
    """
    predictor = get_restock_predictor()
    return predictor.predict_restock_parameters(
        category, demand_forecast, inventory_level, price, units_sold, current_time, current_date
    )

def get_current_restock_parameters(category: str) -> Dict[str, int]:
    """
    Get the current restock parameters for a category.
    
    Args:
        category: Product category
        
    Returns:
        Dictionary with current restock parameters
    """
    predictor = get_restock_predictor()
    return predictor.get_current_parameters(category)

def predict_restock_parameters_once(
    category: str,
    demand_forecast: float,
    inventory_level: float,
    price: float,
    units_sold: float,
    current_date=None
) -> Dict[str, int]:
    """
    Make a one-time prediction of restock parameters using improved models.
    
    The restock level is no longer predicted by the model as it will be
    dynamically calculated as demand_forecast * restock_period in the environment.
    
    Args:
        category: Product category
        demand_forecast: Current demand forecast
        inventory_level: Current inventory level
        price: Current price
        units_sold: Recent units sold
        current_date: Current date (defaults to today)
        
    Returns:
        Dictionary with predicted restock parameters
    """
    predictor = get_restock_predictor()
    
    # Create feature vector with improved features
    X = predictor.prepare_feature_vector(
        category, demand_forecast, inventory_level, price, units_sold, current_date
    )
    
    # Make predictions for amount and period using improved models
    try:
        # Get the amount prediction from EOQ model
        amount = max(10, int(predictor.models['amount'].predict(X)[0]))
        
        # Get the period prediction
        period = max(1, int(predictor.models['period'].predict(X)[0]))
        
        # Return parameters (without level)
        params = {
            'restock_period': period,
            'restock_amount': amount
        }
        
        print(f"One-time restock parameter prediction for {category}:")
        print(f"  Amount: {amount}, Period: {period}")
        print(f"  (Restock level will be calculated dynamically as demand * period)")
        
        return params
    except Exception as e:
        print(f"Error predicting restock parameters: {e}")
        print("Using default values")
        
        # Fallback to reasonable defaults by category
        if category == "Groceries":
            return {'restock_period': 2, 'restock_amount': 150}
        elif category == "Electronics":
            return {'restock_period': 6, 'restock_amount': 50}
        elif category == "Furniture":
            return {'restock_period': 10, 'restock_amount': 30}
        elif category == "Clothing":
            return {'restock_period': 4, 'restock_amount': 80}
        elif category == "Toys":
            return {'restock_period': 4, 'restock_amount': 60}
        else:
            return {'restock_period': 5, 'restock_amount': 100}

if __name__ == "__main__":
    # Test the module
    predictor = RestockPredictor()
    
    # Option 1: Train new models
    try:
        predictor.train_and_save()
        print("Successfully trained and saved restock prediction models")
    except Exception as e:
        print(f"Failed to train models: {e}")
        print("Loading pre-trained models instead...")
        predictor.load_models()
    
    # Test predictions
    categories = ["Electronics", "Furniture", "Clothing", "Toys", "Groceries"]
    for i, category in enumerate(categories):
        # Simulate different time steps
        for time_step in [0, 3, 6, 9, 12]:
            params = predictor.predict_restock_parameters(
                category=category,
                demand_forecast=100,
                inventory_level=50,
                price=100,
                units_sold=20,
                current_time=time_step
            )
            
            if params:
                print(f"\nTime step {time_step}, Category: {category}")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nTime step {time_step}, Category: {category} - No prediction needed yet") 