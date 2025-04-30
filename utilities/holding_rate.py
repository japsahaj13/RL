"""
Holding Rate Calculator Module.

This module provides functionality to calculate dynamic holding cost rates
based on inventory levels and demand forecasts, adapted from the h_rate.py module.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union


def compute_holding_cost_rate(
    csv_path: str,
    unit_cost: float,
    storage_fee: float,
    annual_finance_rate: float,
    periods_per_year: int,
    spoilage_pct: float,
    x0: float,
    anchor_width: float = 0.1,
    r_low_frac: float = 0.2,
    r_high_frac: float = 0.8
) -> float:
    """
    Computes a holding cost rate based on inventory levels and demand forecasts.
    
    The function uses a logistic function to map the average Excess Inventory Rate (EIR)
    to a holding cost rate that varies between 0 and L (maximum rate).
    
    Args:
        csv_path: Path to the CSV file containing inventory data
        unit_cost: Cost per unit of product
        storage_fee: Storage fee per unit per period
        annual_finance_rate: Annual finance rate (e.g., 0.12 for 12%)
        periods_per_year: Number of periods in a year (e.g., 12 for monthly)
        spoilage_pct: Spoilage percentage per period (as a decimal)
        x0: Midpoint of the logistic function (EIR value where rate = L/2)
        anchor_width: Width around x0 for anchor points
        r_low_frac: Fraction of L for the lower anchor point
        r_high_frac: Fraction of L for the higher anchor point
    
    Returns:
        Computed holding cost rate as a float
    """
    # Input validation
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if (unit_cost <= 0 or storage_fee < 0 or annual_finance_rate < 0 or 
            periods_per_year <= 0 or spoilage_pct < 0):
        raise ValueError(
            "Cost and rate parameters must be non-negative and "
            "periods_per_year must be positive"
        )
    
    if not (0 <= x0 <= 1):
        raise ValueError("Midpoint x0 must be between 0 and 1")
    
    if not (0 < anchor_width < 1) or not (0 <= r_low_frac < r_high_frac <= 1):
        raise ValueError(
            "Invalid anchor parameters: ensure 0 < anchor_width < 1 "
            "and 0 <= r_low_frac < r_high_frac <= 1"
        )
    
    # Step 1: Data ingestion & preprocessing
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    inventory_col = 'Inventory Level'
    demand_col = 'Demand Forecast'
    
    # Verify required columns exist
    if inventory_col not in df.columns:
        raise ValueError(f"Required column '{inventory_col}' not found in CSV")
    if demand_col not in df.columns:
        raise ValueError(f"Required column '{demand_col}' not found in CSV")
    
    # Convert to numeric, coerce errors to NaN
    df[inventory_col] = pd.to_numeric(df[inventory_col], errors='coerce')
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    
    # Drop rows where inventory_level is NaN or <= 0
    df = df.dropna(subset=[inventory_col, demand_col])
    df_valid = df[df[inventory_col] > 0].copy()
    
    if df_valid.empty:
        raise ValueError("No valid data rows after preprocessing")
    
    # Optional: Clip extreme outliers at 1st/99th percentiles
    for col in [inventory_col, demand_col]:
        lower = df_valid[col].quantile(0.01)
        upper = df_valid[col].quantile(0.99)
        df_valid[col] = df_valid[col].clip(lower=lower, upper=upper)
    
    # Step 2: Compute per-row Excess Inventory Rate (EIR)
    df_valid['eir'] = df_valid.apply(
        lambda row: max(
            0, 
            (row[inventory_col] - row[demand_col]) / row[inventory_col]
        ),
        axis=1
    )
    
    # Step 3: Aggregate to avg_eir
    avg_eir = df_valid['eir'].mean()
    
    # Step 4: Estimate the maximum holding-cost rate (L) via business rules
    capital_cost = annual_finance_rate * unit_cost / periods_per_year
    spoilage_cost = spoilage_pct * unit_cost
    L = storage_fee + capital_cost + spoilage_cost
    
    # Step 5: Set anchor points based on x0 (not avg_eir)
    EIR_low = max(0, x0 - anchor_width)
    EIR_high = min(1, x0 + anchor_width)
    
    # Ensure anchor points are distinct
    if abs(EIR_high - EIR_low) < 1e-6:
        raise ValueError(
            "Anchor points are too close together. "
            "Increase anchor_width or adjust x0."
        )
    
    r_low = r_low_frac * L
    r_high = r_high_frac * L
    
    # Step 6: Solve for curve steepness (k) using anchor equations
    try:
        # r_low = L / (1 + e^(-k * (EIR_low - x0)))
        # r_high = L / (1 + e^(-k * (EIR_high - x0)))
        # Solve for k:
        ln_low = np.log(L / r_low - 1)
        ln_high = np.log(L / r_high - 1)
        
        k = (ln_low - ln_high) / (EIR_high - EIR_low)
        
        if not np.isfinite(k) or k <= 0:
            raise ValueError("Invalid k value calculated")
            
    except (ValueError, ZeroDivisionError):
        # Fallback to a reasonable default k value
        k = 10.0
    
    # Step 7: Compute final holding_cost_rate using the logistic function
    rate = L / (1 + np.exp(-k * (avg_eir - x0)))
    
    return rate


def compute_max_holding_cost_rate(
    unit_cost: float,
    storage_fee: float,
    annual_finance_rate: float,
    periods_per_year: int,
    spoilage_pct: float
) -> float:
    """
    Computes the maximum possible holding cost rate.
    
    Args:
        unit_cost: Cost per unit of product
        storage_fee: Storage fee per unit per period
        annual_finance_rate: Annual finance rate (e.g., 0.12 for 12%)
        periods_per_year: Number of periods in a year (e.g., 12 for monthly)
        spoilage_pct: Spoilage percentage per period (as a decimal)
    
    Returns:
        Maximum holding cost rate (L)
    """
    capital_cost = annual_finance_rate * unit_cost / periods_per_year
    spoilage_cost = spoilage_pct * unit_cost
    L = storage_fee + capital_cost + spoilage_cost
    return L


def compute_dynamic_holding_cost(
    inventory: float,
    demand_forecast: float,
    unit_cost: float,
    storage_fee: float,
    annual_finance_rate: float,
    periods_per_year: int,
    spoilage_pct: float,
    x0: float = 0.25,
    k: float = 10.0
) -> float:
    """
    Compute a dynamic holding cost based on current inventory and demand forecast.
    
    This function is designed to be used within the RL environment where
    we don't want to read from a CSV file each time.
    
    Args:
        inventory: Current inventory level
        demand_forecast: Forecasted demand
        unit_cost: Cost per unit of product
        storage_fee: Storage fee per unit per period
        annual_finance_rate: Annual finance rate
        periods_per_year: Number of periods in a year
        spoilage_pct: Spoilage percentage per period
        x0: Midpoint of the logistic function
        k: Steepness of the logistic curve
        
    Returns:
        Dynamic holding cost rate
    """
    if inventory <= 0:
        return 0.0
    
    # Calculate Excess Inventory Rate (EIR)
    eir = max(0, (inventory - demand_forecast) / inventory)
    
    # Calculate maximum holding cost rate (L)
    L = compute_max_holding_cost_rate(
        unit_cost=unit_cost,
        storage_fee=storage_fee,
        annual_finance_rate=annual_finance_rate,
        periods_per_year=periods_per_year,
        spoilage_pct=spoilage_pct
    )
    
    # Apply logistic function
    rate = L / (1 + np.exp(-k * (eir - x0)))
    
    return rate


def _run_sanity_checks(L: float, k: float, x0: float):
    """
    Helper function to run sanity checks on the logistic function parameters.
    """
    # Test EIR values
    test_eirs = [0.0, x0 - 0.3, x0 - 0.1, x0, x0 + 0.1, x0 + 0.3, 1.0]
    
    print("\nSanity Check Results:")
    print("-" * 40)
    print(f"Parameters: L={L:.4f}, k={k:.4f}, x0={x0:.4f}")
    print("-" * 40)
    print("EIR\t\tRate\t\t% of L")
    print("-" * 40)
    
    for eir in test_eirs:
        rate = L / (1 + np.exp(-k * (eir - x0)))
        percent_of_L = (rate / L) * 100
        print(f"{eir:.4f}\t\t{rate:.4f}\t\t{percent_of_L:.2f}%")
    
    # Check key properties
    mid_rate = L / (1 + np.exp(-k * (x0 - x0)))
    low_rate = L / (1 + np.exp(-k * (0.0 - x0)))
    high_rate = L / (1 + np.exp(-k * (1.0 - x0)))
    
    assert abs(mid_rate - L/2) < 0.001, f"Rate at x0 should be L/2, got {mid_rate}"
    assert low_rate < L/2, f"Rate at EIR=0 should be < L/2, got {low_rate}"
    assert high_rate > L/2, f"Rate at EIR=1 should be > L/2, got {high_rate}"
    
    print("\nAll sanity checks passed!")


def main():
    """Main function for demonstration and testing."""
    try:
        # Default CSV path
        csv_path = 'retail_store_inventory.csv'
        
        # Example business parameters
        unit_cost = 50.0
        storage_fee = 0.15  # FIXED: Reduced from 2.0 to realistic value
        annual_finance_rate = 0.12
        periods_per_year = 12
        spoilage_pct = 0.002  # FIXED: Reduced from 0.02 to realistic value
        
        # Set x0 as a business threshold (not equal to avg_eir)
        x0 = 0.25  # Midpoint where rate = L/2
        
        # Calculate maximum holding cost rate (L)
        capital_cost = annual_finance_rate * unit_cost / periods_per_year
        spoilage_cost = spoilage_pct * unit_cost
        L = storage_fee + capital_cost + spoilage_cost
        
        # Calculate rate
        rate = compute_holding_cost_rate(
            csv_path=csv_path,
            unit_cost=unit_cost,
            storage_fee=storage_fee,
            annual_finance_rate=annual_finance_rate,
            periods_per_year=periods_per_year,
            spoilage_pct=spoilage_pct,
            x0=x0
        )
        
        # Print results
        print("\nHolding Cost Rate Calculation Results:")
        print("-" * 40)
        print(f"Final Holding Cost Rate: {rate:.4f}")
        print(f"Maximum Holding Cost Rate (L): {L:.4f}")
        print(f"Midpoint (x0): {x0:.4f}")
        
        # Run sanity checks with some typical k value
        _run_sanity_checks(L, 10.0, x0)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
