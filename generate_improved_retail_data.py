#!/usr/bin/env python3
"""
Generate Improved Retail Store Inventory Data

This script generates synthetic retail inventory data with realistic inventory management rules:
1. Explicit inventory management with reorder points and quantities
2. Units Ordered based on demand patterns and inventory levels
3. Restock timing dependent on inventory thresholds
4. EOQ-based order quantities for optimal inventory management
"""

import pandas as pd
import numpy as np
import random
import math
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants for data generation
NUM_STORES = 5
NUM_PRODUCTS_PER_CATEGORY = 20
CATEGORIES = ["Electronics", "Furniture", "Clothing", "Toys", "Groceries"]
REGIONS = ["North", "South", "East", "West"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
DATE_RANGE = 365  # Generate one year of data

def calculate_eoq(annual_demand, order_cost, holding_cost):
    """
    Calculate Economic Order Quantity
    
    Args:
        annual_demand: Annual demand in units
        order_cost: Fixed cost per order
        holding_cost: Annual holding cost per unit
    
    Returns:
        Economic order quantity (minimum 10 units)
    """
    if annual_demand <= 0 or holding_cost <= 0:
        return 100  # Default value if inputs are invalid
    
    eoq = math.sqrt(2 * annual_demand * order_cost / holding_cost)
    return max(10, int(eoq))  # Ensure minimum order size

def generate_improved_retail_data():
    """
    Generate improved retail inventory data with realistic patterns.
    
    Returns:
        DataFrame with retail inventory data
    """
    print("Generating improved retail inventory data...")
    
    # Define base parameters by category
    category_params = {
        "Electronics": {
            "price_range": (50, 500),
            "base_demand_range": (5, 30),
            "price_elasticity": -0.8,  # Higher price = lower demand
            "seasonality": {"Spring": 0.9, "Summer": 0.8, "Autumn": 1.1, "Winter": 1.2},
            "holding_cost_rate": 0.15,  # 15% of item value per year
            "order_cost": 75,  # Fixed cost per order
            "lead_time_days": 3,  # Days between order and delivery
            "safety_stock_factor": 1.5  # Multiple of standard deviation
        },
        "Furniture": {
            "price_range": (100, 1000),
            "base_demand_range": (2, 15),
            "price_elasticity": -0.5,
            "seasonality": {"Spring": 1.2, "Summer": 0.9, "Autumn": 0.8, "Winter": 1.1},
            "holding_cost_rate": 0.12,
            "order_cost": 100,
            "lead_time_days": 5,
            "safety_stock_factor": 1.3
        },
        "Clothing": {
            "price_range": (20, 200),
            "base_demand_range": (10, 50),
            "price_elasticity": -1.2,
            "seasonality": {"Spring": 1.1, "Summer": 1.3, "Autumn": 1.2, "Winter": 0.7},
            "holding_cost_rate": 0.18,
            "order_cost": 50,
            "lead_time_days": 2,
            "safety_stock_factor": 1.4
        },
        "Toys": {
            "price_range": (15, 150),
            "base_demand_range": (8, 40),
            "price_elasticity": -1.0,
            "seasonality": {"Spring": 0.8, "Summer": 1.0, "Autumn": 0.9, "Winter": 1.8},
            "holding_cost_rate": 0.14,
            "order_cost": 60,
            "lead_time_days": 2,
            "safety_stock_factor": 1.2
        },
        "Groceries": {
            "price_range": (2, 50),
            "base_demand_range": (20, 100),
            "price_elasticity": -0.3,
            "seasonality": {"Spring": 1.0, "Summer": 1.1, "Autumn": 1.0, "Winter": 0.9},
            "holding_cost_rate": 0.25,  # Higher due to perishability
            "order_cost": 40,
            "lead_time_days": 1,
            "safety_stock_factor": 1.1
        }
    }
    
    # Initialize empty lists to store data
    data = []
    
    # For each store
    for store_id in range(1, NUM_STORES + 1):
        store_id_str = f"S{store_id:03d}"
        
        # For each category
        for category in CATEGORIES:
            # Assign a region to this store-category combination
            region = random.choice(REGIONS)
            category_param = category_params[category]
            
            # For each product in this category
            for product_id in range(1, NUM_PRODUCTS_PER_CATEGORY + 1):
                product_id_str = f"P{product_id:04d}"
                
                # Set product-specific parameters
                base_price = random.uniform(*category_param["price_range"])
                base_demand = random.uniform(*category_param["base_demand_range"])
                
                # Initialize inventory
                initial_inventory = int(base_demand * 30)  # Start with ~30 days of inventory
                current_inventory = initial_inventory
                
                # Calculate reorder point and EOQ
                # Reorder point = Average demand during lead time + safety stock
                daily_demand_std = base_demand * 0.2  # Assume 20% standard deviation
                safety_stock = daily_demand_std * category_param["safety_stock_factor"]
                reorder_point = (base_demand * category_param["lead_time_days"]) + safety_stock
                
                # Calculate EOQ
                annual_demand = base_demand * 365
                order_cost = category_param["order_cost"]
                holding_cost = category_param["holding_cost_rate"] * base_price
                eoq = calculate_eoq(annual_demand, order_cost, holding_cost)
                
                # Track when we placed orders - to implement lead time
                pending_orders = []  # List of (delivery_date, amount) tuples
                
                # Simulate each day
                start_date = datetime(2022, 1, 1)
                for day in range(DATE_RANGE):
                    current_date = start_date + timedelta(days=day)
                    
                    # Determine season
                    month = current_date.month
                    if month in [3, 4, 5]:
                        season = "Spring"
                    elif month in [6, 7, 8]:
                        season = "Summer"
                    elif month in [9, 10, 11]:
                        season = "Autumn"
                    else:
                        season = "Winter"
                    
                    # Random weather
                    weather = random.choice(WEATHER_CONDITIONS)
                    
                    # Apply weather effects
                    weather_effects = {
                        "Sunny": 1.1,
                        "Cloudy": 1.0,
                        "Rainy": 0.9,
                        "Snowy": 0.7
                    }
                    
                    # Determine if it's a promotion day (weekends and some random days)
                    is_promotion = 1 if current_date.weekday() >= 5 or random.random() < 0.1 else 0
                    promotion_effect = 1.3 if is_promotion else 1.0
                    
                    # Set price (with slight variations and occasional discounts)
                    discount_pct = random.choice([0, 0, 0, 10, 20, 30]) if is_promotion else 0
                    price = base_price * (1 - discount_pct / 100) * (0.95 + 0.1 * random.random())
                    
                    # Competitor pricing (correlated with our price but with variations)
                    competitor_price = price * (0.9 + 0.2 * random.random())
                    
                    # Calculate demand based on price, season, weather, etc.
                    price_effect = (price / base_price) ** category_param["price_elasticity"]
                    seasonal_effect = category_param["seasonality"][season]
                    weather_effect = weather_effects[weather]
                    
                    # Daily demand with some randomness
                    daily_demand = (base_demand * price_effect * seasonal_effect * 
                                   weather_effect * promotion_effect)
                    
                    # Add random noise (normal distribution around the calculated demand)
                    noise_factor = max(0.5, np.random.normal(1.0, 0.2))
                    daily_demand = daily_demand * noise_factor
                    
                    # Round to integer
                    daily_demand = max(0, int(round(daily_demand)))
                    
                    # Calculate demand forecast (slightly different from actual demand)
                    forecast_error = np.random.normal(1.0, 0.15)  # 15% error standard deviation
                    demand_forecast = max(1, int(round(daily_demand * forecast_error)))
                    
                    # Process any pending orders arriving today
                    units_received = 0
                    new_pending_orders = []
                    for delivery_date, amount in pending_orders:
                        if delivery_date <= current_date:
                            current_inventory += amount
                            units_received += amount
                        else:
                            new_pending_orders.append((delivery_date, amount))
                    pending_orders = new_pending_orders
                    
                    # Calculate actual sales (limited by inventory)
                    units_sold = min(daily_demand, current_inventory)
                    current_inventory -= units_sold
                    
                    # Check if we need to place a new order
                    units_ordered = 0
                    if current_inventory <= reorder_point:
                        # Calculate how much to order based on EOQ
                        units_ordered = eoq
                        
                        # Add to pending orders (will arrive after lead time)
                        delivery_date = current_date + timedelta(days=category_param["lead_time_days"])
                        pending_orders.append((delivery_date, units_ordered))
                    
                    # Prepare row data
                    row = {
                        "Date": current_date.strftime("%Y-%m-%d"),
                        "Store ID": store_id_str,
                        "Product ID": product_id_str,
                        "Category": category,
                        "Region": region,
                        "Inventory Level": current_inventory,
                        "Units Sold": units_sold,
                        "Units Ordered": units_ordered,
                        "Units Received": units_received,
                        "Demand Forecast": demand_forecast,
                        "Actual Demand": daily_demand,
                        "Price": round(price, 2),
                        "Discount": discount_pct,
                        "Weather Condition": weather,
                        "Holiday/Promotion": is_promotion,
                        "Competitor Pricing": round(competitor_price, 2),
                        "Seasonality": season,
                        "Reorder Point": int(reorder_point),
                        "EOQ": eoq,
                        "Lead Time Days": category_param["lead_time_days"],
                        "Safety Stock": int(safety_stock),
                        "log_price": np.log(price),
                        "log_competitor_price": np.log(competitor_price)
                    }
                    
                    data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records")
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_improved_retail_data()
    
    # Save to CSV
    output_path = "data/improved_retail_store_inventory.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved improved retail data to {output_path}")
    
    # Print sample
    print("\nSample data:")
    print(df.head())
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(df[["Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast", "Actual Demand"]].describe())
    
    # Print restock statistics
    restock_events = df[df["Units Ordered"] > 0]
    print(f"\nTotal restock events: {len(restock_events)}")
    print("Average restock amount by category:")
    print(restock_events.groupby("Category")["Units Ordered"].mean()) 