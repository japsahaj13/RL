"""
Demand Model Fitting Module.

This module fits log-log demand models from historical retail data,
extracting parameters for use in the MSME environment simulation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def fit_demand_models(
    csv_path: str = 'data/retail_store_inventory.csv',
    enhanced_csv_path: str = 'data/enhanced_retail_store_inventory.csv',
    save_path: str = 'demand_models.pkl'
) -> Dict[str, Dict[str, Any]]:
    """
    Fit log-log demand models from historical retail data.
    
    Args:
        csv_path: Path to the CSV file with retail data
        enhanced_csv_path: Path to the CSV file with enhanced retail data
        save_path: Path to save the fitted models
        
    Returns:
        Dictionary of parameters for each product category
    """
    print(f"Fitting demand models from data...")
    
    # Try to load enhanced data first
    try:
        df = pd.read_csv(enhanced_csv_path)
        print(f"Using enhanced dataset from {enhanced_csv_path}")
    except FileNotFoundError:
        try:
            df = pd.read_csv(csv_path)
            print(f"Using standard dataset from {csv_path}")
        except FileNotFoundError:
            print("No dataset found. Using default parameters...")
            return _create_default_parameters()
    
    # Create models dictionary to store parameters
    models = {}
    
    # Get unique product categories
    categories = df['Category'].unique()
    
    print(f"Fitting models for {len(categories)} product categories...")
    
    # For each category, fit a log-log demand model
    for category in categories:
        print(f"  - Fitting model for {category}...")
        
        # Filter data for this category
        category_data = df[df['Category'] == category].copy()
        
        # Apply log transformation to relevant columns
        # We need to avoid log(0), so add a small constant
        epsilon = 1e-6
        category_data['log_price'] = np.log(category_data['Price'] + epsilon)
        category_data['log_competitor_price'] = np.log(category_data['Competitor Pricing'] + epsilon)
        
        # Use Units Sold as the demand
        if 'Demand' in category_data.columns:
            category_data['log_demand'] = np.log(category_data['Demand'] + epsilon)
        else:
            category_data['log_demand'] = np.log(category_data['Units Sold'] + epsilon)
        
        # Fit log-log model: log(D) = log(D0) + e*log(P) + c*log(P_comp)
        X = category_data[['log_price', 'log_competitor_price']]
        y = category_data['log_demand']
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients
        price_elasticity = model.coef_[0]  # e
        competitor_sensitivity = model.coef_[1]  # c
        intercept = model.intercept_  # log(D0)
        
        # Calculate base demand
        base_demand = math.exp(intercept)
        
        # Compute R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Calculate season effects
        season_effects = _calculate_effects(category_data, 'Seasonality', 'Demand' if 'Demand' in category_data.columns else 'Units Sold')
        
        # Calculate weather effects
        weather_effects = _calculate_effects(category_data, 'Weather Condition', 'Demand' if 'Demand' in category_data.columns else 'Units Sold')
        
        # Calculate region effects
        region_effects = _calculate_effects(category_data, 'Region', 'Demand' if 'Demand' in category_data.columns else 'Units Sold')
        
        # Calculate promotion multiplier
        if 'Holiday/Promotion' in category_data.columns:
            promo_data = category_data[category_data['Holiday/Promotion'] == 1]
            non_promo_data = category_data[category_data['Holiday/Promotion'] == 0]
            
            if len(promo_data) > 0 and len(non_promo_data) > 0:
                target_col = 'Demand' if 'Demand' in category_data.columns else 'Units Sold'
                avg_promo_demand = promo_data[target_col].mean()
                avg_non_promo_demand = non_promo_data[target_col].mean()
                promotion_multiplier = avg_promo_demand / avg_non_promo_demand if avg_non_promo_demand > 0 else 1.2
            else:
                promotion_multiplier = 1.2  # Default
        else:
            promotion_multiplier = 1.2  # Default
        
        # Store parameters
        models[category] = {
            'base_demand': base_demand,
            'price_elasticity': price_elasticity,
            'competitor_sensitivity': competitor_sensitivity,
            'promotion_multiplier': promotion_multiplier,
            'season_effects': season_effects,
            'weather_effects': weather_effects,
            'region_effects': region_effects,
            'r2': r2
        }
        
        print(f"    R² = {r2:.3f}, Price Elasticity = {price_elasticity:.3f}, Competitor Sensitivity = {competitor_sensitivity:.3f}")
    
    # Save models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Models saved to {save_path}")
    return models

def _calculate_effects(
    df: pd.DataFrame, 
    factor_column: str, 
    target_column: str
) -> Dict[str, float]:
    """
    Calculate multiplier effects for a categorical factor.
    
    Args:
        df: DataFrame with the data
        factor_column: Column with the categorical factor
        target_column: Column with the target variable
        
    Returns:
        Dictionary of effects for each factor level
    """
    if factor_column not in df.columns or target_column not in df.columns:
        # Return default values if columns don't exist
        if factor_column == 'Seasonality':
            return {'Spring': 1.0, 'Summer': 1.2, 'Autumn': 0.8, 'Winter': 0.9}
        elif factor_column == 'Weather Condition':
            return {'Sunny': 1.0, 'Cloudy': 0.9, 'Rainy': 0.8, 'Snowy': 0.7}
        elif factor_column == 'Region':
            return {'North': 1.0, 'South': 1.1, 'East': 0.9, 'West': 1.0}
        else:
            return {}
    
    # Get average target value
    overall_avg = df[target_column].mean()
    if overall_avg <= 0:
        overall_avg = 1.0  # Fallback to prevent division by zero
    
    # Calculate effect for each factor level
    effects = {}
    for level in df[factor_column].unique():
        level_data = df[df[factor_column] == level]
        if len(level_data) > 0:
            level_avg = level_data[target_column].mean()
            # Calculate effect as ratio to overall average
            effect = level_avg / overall_avg if overall_avg > 0 else 1.0
            effects[level] = effect
        else:
            effects[level] = 1.0  # Default if no data
    
    return effects

def _create_default_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Create default parameters when no data is available.
    
    Returns:
        Dictionary of default parameters for each product category
    """
    print("Creating default parameters...")
    
    # Default parameters
    default_params = {
        'Electronics': {
            'base_demand': 200,
            'price_elasticity': -0.8, 
            'competitor_sensitivity': 0.3,
            'promotion_multiplier': 1.2,
            'season_effects': {'Spring': 1.0, 'Summer': 1.2, 'Autumn': 0.8, 'Winter': 0.9},
            'weather_effects': {'Sunny': 1.0, 'Cloudy': 0.9, 'Rainy': 0.8, 'Snowy': 0.7},
            'region_effects': {'North': 1.0, 'South': 1.1, 'East': 0.9, 'West': 1.0}
        },
        'Groceries': {
            'base_demand': 350,
            'price_elasticity': -1.2,
            'competitor_sensitivity': 0.4,
            'promotion_multiplier': 1.5,
            'season_effects': {'Spring': 1.0, 'Summer': 1.1, 'Autumn': 0.9, 'Winter': 1.0},
            'weather_effects': {'Sunny': 1.0, 'Cloudy': 0.95, 'Rainy': 0.9, 'Snowy': 0.8},
            'region_effects': {'North': 0.9, 'South': 1.2, 'East': 0.9, 'West': 1.0}
        },
        'Toys': {
            'base_demand': 180,
            'price_elasticity': -0.9,
            'competitor_sensitivity': 0.25,
            'promotion_multiplier': 1.4,
            'season_effects': {'Spring': 0.9, 'Summer': 0.8, 'Autumn': 1.0, 'Winter': 1.3},
            'weather_effects': {'Sunny': 0.9, 'Cloudy': 1.0, 'Rainy': 1.1, 'Snowy': 1.2},
            'region_effects': {'North': 1.0, 'South': 1.0, 'East': 1.0, 'West': 1.0}
        },
        'Furniture': {
            'base_demand': 120,
            'price_elasticity': -0.7,
            'competitor_sensitivity': 0.2,
            'promotion_multiplier': 1.3,
            'season_effects': {'Spring': 1.1, 'Summer': 1.0, 'Autumn': 1.0, 'Winter': 0.9},
            'weather_effects': {'Sunny': 1.0, 'Cloudy': 1.0, 'Rainy': 0.9, 'Snowy': 0.8},
            'region_effects': {'North': 0.9, 'South': 1.1, 'East': 1.0, 'West': 1.0}
        },
        'Clothing': {
            'base_demand': 250,
            'price_elasticity': -1.0,
            'competitor_sensitivity': 0.35,
            'promotion_multiplier': 1.3,
            'season_effects': {'Spring': 1.1, 'Summer': 0.9, 'Autumn': 1.2, 'Winter': 0.8},
            'weather_effects': {'Sunny': 1.1, 'Cloudy': 1.0, 'Rainy': 0.9, 'Snowy': 0.8},
            'region_effects': {'North': 0.9, 'South': 1.1, 'East': 1.0, 'West': 1.0}
        }
    }
    
    return default_params

if __name__ == "__main__":
    # When run as a script, fit models and save to file
    fit_demand_models() 