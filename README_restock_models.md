# Restock Parameter Prediction Models

This document explains the restock parameter prediction system used in the RL retail pricing project. The system uses machine learning models to predict optimal restock parameters based on historical retail data.

## Overview

The restock prediction system consists of three RandomForest regression models that predict:

1. **Restock Level** - The inventory level at which to trigger a restock
2. **Restock Amount** - How many units to order during a restock
3. **Restock Period** - How many days to wait between restock checks

These models are trained on historical retail inventory data and are used during simulation to make the environment more realistic.

## Getting Started

### Prerequisites

- Python 3.8+
- Scikit-learn
- Pandas
- NumPy
- Required retail inventory data in `data/retail_store_inventory.csv`

### Training the Models

After cloning the repository, you need to train the restock prediction models before running the RL simulations. To do this, run:

```bash
python train_restock_models.py
```

This script will:
1. Load and preprocess the retail inventory data
2. Extract features and targets for each parameter
3. Train RandomForest models for each parameter
4. Save the models to the `models/` directory
5. Print a summary of model performance

The trained models will be saved as:
- `models/restock_level_model.pkl`
- `models/restock_amount_model.pkl`
- `models/restock_period_model.pkl`

## How the Models Work

### Data Inputs

The models use the following features to make predictions:

- Product category (one-hot encoded)
- Current demand forecast
- Current inventory level
- Current price
- Recent units sold
- Day of week and month (to capture seasonality)

### Model Architecture

Each model is a scikit-learn Pipeline with:
1. **StandardScaler** - To normalize input features
2. **RandomForestRegressor** - To predict the parameter value

### Integration with Environment

The models are used by the `MSMEEnvironment` class through the `RestockPredictor` in `utilities/restock_prediction.py`. The prediction workflow is:

1. The environment tracks the time since the last restock check
2. When it's time for a new check, it calls `predict_restock_parameters()`
3. The function uses the current state (inventory, demand, etc.) to make predictions
4. The environment updates its restock parameters for future restocking decisions

## Configuration

To enable dynamic restock prediction, set the `use_dynamic_restock` parameter to `True` in your environment configuration:

```python
config = MSMEConfig(
    # Other parameters...
    use_dynamic_restock=True
)
```

## Integrating with RL Training

The optimized reward weights (from hyperparameter tuning) are used in the training scripts:

```python
env = MSMEEnvironment(
    config=config,
    # Use optimized reward weights from hyperparameter tuning
    alpha=0.1319,  # Revenue weight
    beta=0.2316,   # Market share weight
    gamma=0.3654,  # Inventory/price stability weight
    delta=0.2712   # Profit margin weight
)
```

These reward weights prioritize price stability and profit margin over simply following competitor prices, which helps create more realistic MSME behavior.

## Troubleshooting

If you encounter issues with the restock prediction models:

1. Check that the retail data file exists at `data/retail_store_inventory.csv`
2. Verify that the models were successfully trained and saved in the `models/` directory
3. Check the `restock_model_training.log` file for error messages
4. Ensure the feature columns match between training and inference

## Technical Details

The restock prediction models:
- Use RandomForest regression with 100 estimators
- Are trained on approximately 73,100 restock events
- Have separate models for each parameter (level, amount, period)
- Save the full preprocessing pipeline with the model
- Calculate feature importance to understand prediction factors

Models only predict new parameters when the previous restock period has been exhausted, which prevents overly frequent parameter changes. 