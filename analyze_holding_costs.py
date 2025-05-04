#!/usr/bin/env python3
"""
Analyze holding costs in retail datasets and compare with industry standards
"""

import pandas as pd
import numpy as np

# Industry standard holding cost rates (annual percentage of inventory value)
INDUSTRY_STANDARDS = {
    'General Retail': (15, 30),  # 15-30% annual range
    'Electronics': (15, 25),     # Electronics typically 15-25%
    'Furniture': (10, 20),       # Furniture typically 10-20%
    'Clothing': (15, 30),        # Apparel typically 15-30%
    'Toys': (12, 25),            # Toys typically 12-25%
    'Groceries': (20, 35)        # Groceries/perishables typically 20-35%
}

# Holding cost percentages used in the improved dataset
IMPROVED_DATASET_RATES = {
    'Electronics': 15.0,  # 15% of item value per year
    'Furniture': 12.0,    # 12% of item value per year
    'Clothing': 18.0,     # 18% of item value per year
    'Toys': 14.0,         # 14% of item value per year
    'Groceries': 25.0     # 25% of item value per year (higher due to perishability)
}

def calculate_holding_costs(dataset_path):
    """Calculate and analyze holding costs from the dataset"""
    print(f"Analyzing holding costs in {dataset_path}...")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    results = {}
    
    # For each category
    for category in df['Category'].unique():
        # Get subset of data for this category
        cat_df = df[df['Category'] == category]
        
        # Calculate average price and inventory
        avg_price = cat_df['Price'].mean()
        avg_inventory = cat_df['Inventory Level'].mean()
        
        # Get annual holding cost percentage for this category
        annual_rate = IMPROVED_DATASET_RATES.get(category, 20.0)  # Default to 20% if not found
        
        # Calculate daily holding cost
        daily_holding_cost = (annual_rate / 100) * avg_price * avg_inventory / 365
        
        # Calculate annual holding cost
        annual_holding_cost = daily_holding_cost * 365
        
        # Calculate holding cost as percentage of inventory value
        inventory_value = avg_price * avg_inventory
        holding_pct_of_inventory = (annual_holding_cost / inventory_value) * 100
        
        # Get industry standard range
        industry_min, industry_max = INDUSTRY_STANDARDS.get(category, INDUSTRY_STANDARDS['General Retail'])
        
        # Determine if holding cost is within industry standards
        within_standards = industry_min <= annual_rate <= industry_max
        
        # Store results
        results[category] = {
            'avg_price': avg_price,
            'avg_inventory': avg_inventory,
            'inventory_value': inventory_value,
            'annual_rate_pct': annual_rate,
            'daily_holding_cost': daily_holding_cost,
            'annual_holding_cost': annual_holding_cost,
            'holding_pct_of_inventory': holding_pct_of_inventory,
            'industry_standard_range': (industry_min, industry_max),
            'within_standards': within_standards
        }
    
    return results

def print_analysis(results):
    """Print the analysis results in a readable format"""
    print("\nHolding Cost Analysis Results:")
    print("=" * 80)
    print(f"{'Category':<12} {'Avg Price':<10} {'Avg Inv':<10} {'Annual Rate':<12} {'Daily Cost':<12} {'Industry Range':<15} {'Status'}")
    print("-" * 80)
    
    for category, data in results.items():
        status = "✓ Within range" if data['within_standards'] else "✗ Outside range"
        industry_range = f"{data['industry_standard_range'][0]}-{data['industry_standard_range'][1]}%"
        
        print(f"{category:<12} ${data['avg_price']:<9.2f} {data['avg_inventory']:<10.1f} " +
              f"{data['annual_rate_pct']}%{'':<7} ${data['daily_holding_cost']:<11.2f} " +
              f"{industry_range:<15} {status}")
    
    print("=" * 80)
    print("Note: Industry standard range is the annual holding cost as percentage of inventory value")
    print("Holding costs include storage, capital costs, insurance, taxes, and risk of obsolescence")

if __name__ == "__main__":
    # Analyze improved retail dataset
    results = calculate_holding_costs("data/improved_retail_store_inventory.csv")
    print_analysis(results)
    
    # Also check if original retail dataset exists and analyze it for comparison
    try:
        orig_results = calculate_holding_costs("data/retail_store_inventory.csv")
        print("\nComparison with original retail dataset:")
        print_analysis(orig_results)
    except Exception as e:
        print(f"\nCould not analyze original dataset: {e}") 