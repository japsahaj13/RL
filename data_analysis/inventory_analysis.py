"""
Inventory Analysis Module

This module analyzes retail store inventory data to extract data-driven
restock parameters for different product categories.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Make sure the output directory exists
os.makedirs('data_analysis/figures', exist_ok=True)

def load_inventory_data(file_path: str = 'data/retail_store_inventory.csv') -> pd.DataFrame:
    """
    Load and preprocess inventory data.
    
    Args:
        file_path: Path to the inventory data CSV
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading inventory data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter out rows with missing values in key columns
    key_columns = ['Category', 'Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast']
    df = df.dropna(subset=key_columns)
    
    print(f"Loaded data with {len(df):,} records.")
    print(f"Categories: {', '.join(df['Category'].unique())}")
    
    return df

def analyze_restock_patterns(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze restock patterns by category.
    
    Args:
        df: Preprocessed inventory DataFrame
        
    Returns:
        Dictionary with restock parameters by category
    """
    # Group by Category, Store ID, and Product ID
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
        # Filter out NaN values and convert to integers
        periods = [p for p in data['periods'] if not np.isnan(p)]
        levels = [l for l in data['levels'] if not np.isnan(l)]
        amounts = [a for a in data['amounts'] if not np.isnan(a)]
        
        if not periods or not levels or not amounts:
            continue
        
        # Calculate means
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
    
    return category_params

def visualize_restock_parameters(params: Dict[str, Dict[str, float]]) -> None:
    """
    Visualize the extracted restock parameters.
    
    Args:
        params: Dictionary with restock parameters by category
    """
    categories = list(params.keys())
    
    # Extract parameters
    periods = [params[cat]['restock_period'] for cat in categories]
    levels = [params[cat]['restock_level'] for cat in categories]
    amounts = [params[cat]['restock_amount'] for cat in categories]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot restock periods
    axes[0].bar(categories, periods)
    axes[0].set_title('Restock Period by Category (Days)')
    axes[0].set_ylabel('Days')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot restock levels
    axes[1].bar(categories, levels)
    axes[1].set_title('Restock Level by Category (Units)')
    axes[1].set_ylabel('Units')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot restock amounts
    axes[2].bar(categories, amounts)
    axes[2].set_title('Restock Amount by Category (Units)')
    axes[2].set_ylabel('Units')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data_analysis/figures/restock_parameters_by_category.png')
    print(f"Visualization saved to data_analysis/figures/restock_parameters_by_category.png")

def analyze_initial_inventory(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze appropriate initial inventory levels by category.
    
    Args:
        df: Preprocessed inventory DataFrame
        
    Returns:
        Dictionary with initial inventory by category
    """
    # Get the earliest record for each product
    df_sorted = df.sort_values(['Category', 'Store ID', 'Product ID', 'Date'])
    first_records = df_sorted.groupby(['Category', 'Store ID', 'Product ID']).first().reset_index()
    
    # Calculate median initial inventory by category
    category_inventory = {}
    
    for category, group in first_records.groupby('Category'):
        median_inventory = int(round(group['Inventory Level'].median()))
        category_inventory[category] = max(20, median_inventory)
    
    return category_inventory

def main():
    """Main execution function."""
    # Load and preprocess data
    df = load_inventory_data()
    
    # Analyze restock patterns
    print("\nAnalyzing restock patterns by category...")
    restock_params = analyze_restock_patterns(df)
    
    # Analyze initial inventory
    print("\nAnalyzing initial inventory levels by category...")
    initial_inventory = analyze_initial_inventory(df)
    
    # Combine parameters
    all_params = {}
    
    for category in restock_params:
        all_params[category] = {
            **restock_params[category],
            'initial_inventory': initial_inventory.get(category, 100)
        }
    
    # Print results
    print("\nCategory-Specific Inventory Parameters:")
    print("=" * 80)
    
    for category, params in all_params.items():
        print(f"\nCategory: {category}")
        print(f"  Initial Inventory: {params['initial_inventory']} units")
        print(f"  Restock Level: {params['restock_level']} units")
        print(f"  Restock Amount: {params['restock_amount']} units")
        print(f"  Restock Period: {params['restock_period']} days")
        print(f"  Sample Size: {params['sample_size']} product-store combinations")
    
    # Visualize the results
    visualize_restock_parameters(restock_params)
    
    # Generate code for config.py
    print("\nGenerated code for config.py:")
    print("=" * 80)
    
    for category in all_params:
        function_name = f"create_{category.lower()}_config"
        
        print(f"""
def {function_name}() -> MSMEConfig:
    \"\"\"
    Create a configuration for the {category} category
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for {category}
    \"\"\"
    return MSMEConfig(
        product_name="{category} Item",
        product_category="{category}",
        region="North",
        unit_cost=40,
        initial_inventory={all_params[category]['initial_inventory']},
        restock_level={all_params[category]['restock_level']},
        restock_amount={all_params[category]['restock_amount']},
        restock_period={all_params[category]['restock_period']}
    )
        """)

if __name__ == "__main__":
    main() 