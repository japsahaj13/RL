"""
Data-Driven Parameters Module

This module provides functions to extract economic parameters from retail data,
following the principle that base economic parameters should be inferred from
the dataset rather than hardcoded or learned during training.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache for dataset parameters
_parameter_cache = {}

def load_retail_data(file_path: str = 'data/retail_store_inventory.csv') -> pd.DataFrame:
    """
    Load and preprocess the retail inventory dataset.
    
    Args:
        file_path: Path to the retail inventory data
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading retail data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Retail data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter out rows with missing key values
    key_columns = ['Category', 'Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast']
    df = df.dropna(subset=key_columns)
    
    logger.info(f"Loaded {len(df):,} records with categories: {', '.join(df['Category'].unique())}")
    
    return df

def _extract_restock_parameters(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Extract restock parameters from the dataset.
    
    Args:
        df: Preprocessed retail DataFrame
        
    Returns:
        Dictionary with restock parameters by category
    """
    # Check if we have these parameters in cache
    cache_key = "restock_parameters"
    if cache_key in _parameter_cache:
        return _parameter_cache[cache_key]
    
    logger.info("Extracting restock parameters from data...")
    
    # Group by category, store, and product
    store_product_groups = df.groupby(['Category', 'Store ID', 'Product ID'])
    
    # Track restock patterns
    restock_data = {}
    
    for (category, store_id, product_id), group in store_product_groups:
        # Sort by date
        group = group.sort_values('Date')
        
        # Find restock events (when Units Ordered > 0)
        restock_events = group[group['Units Ordered'] > 0].copy()
        
        if len(restock_events) < 2:
            continue
        
        # Calculate restock periods (days between restocks)
        restock_events['Next Restock Date'] = restock_events['Date'].shift(-1)
        restock_events['Days Between Restocks'] = (restock_events['Next Restock Date'] - restock_events['Date']).dt.days
        
        # Calculate inventory levels at time of restock
        restock_events['Inventory Before Restock'] = restock_events['Inventory Level'] - restock_events['Units Ordered']
        
        # Get average values
        avg_restock_period = restock_events['Days Between Restocks'].mean()
        avg_restock_level = restock_events['Inventory Before Restock'].mean()
        avg_restock_amount = restock_events['Units Ordered'].mean()
        
        # Add to category data
        if category not in restock_data:
            restock_data[category] = {
                'periods': [],
                'levels': [],
                'amounts': []
            }
        
        restock_data[category]['periods'].append(avg_restock_period)
        restock_data[category]['levels'].append(avg_restock_level)
        restock_data[category]['amounts'].append(avg_restock_amount)
    
    # Calculate category-level averages
    category_params = {}
    
    for category, data in restock_data.items():
        # Filter out NaN values
        periods = [p for p in data['periods'] if not np.isnan(p)]
        levels = [l for l in data['levels'] if not np.isnan(l)]
        amounts = [a for a in data['amounts'] if not np.isnan(a)]
        
        if not periods or not levels or not amounts:
            continue
        
        # Calculate means and convert to integers
        avg_period = int(round(np.mean(periods)))
        avg_level = int(round(np.mean(levels)))
        avg_amount = int(round(np.mean(amounts)))
        
        # Ensure minimum values
        avg_period = max(1, avg_period)
        avg_level = max(5, avg_level)
        avg_amount = max(10, avg_amount)
        
        category_params[category] = {
            'restock_period': avg_period,
            'restock_level': avg_level,
            'restock_amount': avg_amount,
            'sample_size': len(periods)
        }
    
    # Store in cache
    _parameter_cache[cache_key] = category_params
    
    return category_params

def _extract_initial_inventory(df: pd.DataFrame) -> Dict[str, int]:
    """
    Extract initial inventory levels from the dataset.
    
    Args:
        df: Preprocessed retail DataFrame
        
    Returns:
        Dictionary with initial inventory by category
    """
    # Check if we have these parameters in cache
    cache_key = "initial_inventory"
    if cache_key in _parameter_cache:
        return _parameter_cache[cache_key]
    
    logger.info("Extracting initial inventory levels from data...")
    
    # Get the earliest record for each product
    df_sorted = df.sort_values(['Category', 'Store ID', 'Product ID', 'Date'])
    first_records = df_sorted.groupby(['Category', 'Store ID', 'Product ID']).first().reset_index()
    
    # Calculate median initial inventory by category
    category_inventory = {}
    
    for category, group in first_records.groupby('Category'):
        median_inventory = int(round(group['Inventory Level'].median()))
        category_inventory[category] = max(20, median_inventory)  # Ensure minimum 20 units
    
    # Store in cache
    _parameter_cache[cache_key] = category_inventory
    
    return category_inventory

def _extract_unit_costs(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract average unit costs by category.
    
    Args:
        df: Preprocessed retail DataFrame
        
    Returns:
        Dictionary with unit costs by category
    """
    # For this implementation, we'll use a simple heuristic:
    # Unit cost is approximately 60% of the average price
    
    # Check if we have these parameters in cache
    cache_key = "unit_costs"
    if cache_key in _parameter_cache:
        return _parameter_cache[cache_key]
    
    logger.info("Estimating unit costs from pricing data...")
    
    # Calculate average price by category
    avg_prices = df.groupby('Category')['Price'].mean()
    
    # Estimate unit cost as 60% of average price
    unit_costs = {}
    for category, avg_price in avg_prices.items():
        unit_costs[category] = round(avg_price * 0.6, 2)
    
    # Store in cache
    _parameter_cache[cache_key] = unit_costs
    
    return unit_costs

def get_restock_parameters(category: str) -> Tuple[int, int, int, int]:
    """
    Get data-driven restock parameters for a specific category.
    
    Args:
        category: Product category name
        
    Returns:
        Tuple of (initial_inventory, restock_level, restock_amount, restock_period)
    """
    # Load data if we haven't already
    if not _parameter_cache:
        df = load_retail_data()
        _extract_restock_parameters(df)
        _extract_initial_inventory(df)
    
    # Get from cache
    restock_params = _parameter_cache.get("restock_parameters", {})
    initial_inventory = _parameter_cache.get("initial_inventory", {})
    
    # Default values if category not found
    if category not in restock_params:
        logger.warning(f"No restock data found for category '{category}'. Using defaults.")
        return (80, 30, 100, 6)  # Default values
    
    # Get category-specific values
    params = restock_params[category]
    init_inv = initial_inventory.get(category, 100)
    
    return (
        init_inv,
        params['restock_level'],
        params['restock_amount'],
        params['restock_period']
    )

def get_unit_cost(category: str) -> float:
    """
    Get data-driven unit cost for a specific category.
    
    Args:
        category: Product category name
        
    Returns:
        Estimated unit cost
    """
    # Load data if we haven't already
    if "unit_costs" not in _parameter_cache:
        df = load_retail_data()
        _extract_unit_costs(df)
    
    # Get from cache
    unit_costs = _parameter_cache.get("unit_costs", {})
    
    # Default values based on category if not found
    default_costs = {
        "Electronics": 75.0,
        "Furniture": 150.0,
        "Clothing": 40.0,
        "Toys": 25.0,
        "Groceries": 10.0
    }
    
    return unit_costs.get(category, default_costs.get(category, 40.0))

def get_all_parameters(category: str) -> Dict[str, Any]:
    """
    Get all data-driven parameters for a specific category.
    
    Args:
        category: Product category name
        
    Returns:
        Dictionary with all parameters
    """
    # Load all parameters
    if not _parameter_cache:
        df = load_retail_data()
        _extract_restock_parameters(df)
        _extract_initial_inventory(df)
        _extract_unit_costs(df)
    
    # Get restock parameters
    initial_inv, restock_level, restock_amount, restock_period = get_restock_parameters(category)
    
    # Get unit cost
    unit_cost = get_unit_cost(category)
    
    return {
        "initial_inventory": initial_inv,
        "restock_level": restock_level,
        "restock_amount": restock_amount,
        "restock_period": restock_period,
        "unit_cost": unit_cost
    }

if __name__ == "__main__":
    # Test the module
    for category in ["Electronics", "Furniture", "Clothing", "Toys", "Groceries"]:
        params = get_all_parameters(category)
        print(f"\nCategory: {category}")
        for key, value in params.items():
            print(f"  {key}: {value}") 