# Dynamic Restock Parameter Prediction

This module provides a machine learning approach to predict optimal inventory restock parameters based on historical retail data.

## Overview

The restock prediction model follows the principle that "base economic parameters are not learned from the environment but inferred or estimated from the dataset." 

Key features:
- Pre-trained models to predict restock level, amount, and period
- Parameters are only updated when the previous restock period is exhausted
- Uses retail data demand forecasts and inventory patterns for predictions
- Maintains separation between economic parameter determination and RL agent learning

## How It Works

1. **Model Training (Pre-RL)**
   - Historical restock events are identified in retail_store_inventory.csv
   - Features extracted: demand forecasts, inventory levels, price, seasonality
   - Three separate RandomForest models trained for:
     - Restock Level (when to reorder)
     - Restock Amount (how much to order)
     - Restock Period (time between restock checks)
   
2. **Prediction During Simulation**
   - Initial parameters predicted at start of episode
   - New predictions made only after the previous restock period is exhausted
   - Predictions use current state: inventory, demand forecast, price, etc.
   - Models encapsulate economic relationships from the data

3. **Integration with Environment**
   - Parameters provided to environment through config.get_restock_parameters()
   - Transparent to the RL agent, which only learns pricing strategy
   - Consistent with how competitor price modeling works

## Implementation Details

### Key Files
- `utilities/restock_prediction.py`: Core prediction models and interface
- `environments/config.py`: Integration with configuration system
- `environments/msme_env.py`: Usage in environment restock decisions

### Usage
Enable dynamic restock prediction by setting:
```python
config = MSMEConfig(
    # Other parameters
    use_dynamic_restock=True
)
```

## Benefits

1. **More Realistic Simulation**
   - Parameters adapt to changing market conditions
   - Different product categories have appropriate parameters
   - Mirrors real-world inventory management decisions

2. **Data-Driven Approach**
   - Parameters extracted from retail data rather than hardcoded
   - Captures realistic relationships between demand, inventory, and restock decisions
   - Maintains consistency with real-world retail operations

3. **Improved RL Training**
   - Agent can focus on pricing strategy without handling inventory management
   - More realistic environment leads to better transferable pricing policies
   - Clear separation between economic parameter inference and policy learning 