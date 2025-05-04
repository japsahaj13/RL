"""
Competitor Price Modeling Module.

This module analyzes competitor pricing patterns from historical data
and builds a model to predict competitor pricing behavior.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def fit_competitor_price_models(
        csv_path: str = 'data/retail_store_inventory.csv',
        save_path: str = 'models/saved/competitor_models.pkl',
        use_advanced_model: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Fit competitor pricing models from historical retail data.
    
    Args:
        csv_path: Path to the CSV file with retail data
        save_path: Path to save the fitted models
        use_advanced_model: Whether to use RandomForest (True) or LinearRegression (False)
        
    Returns:
        Dictionary of fitted models for each product category
    """
    print(f"Loading data from {csv_path}...")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
    
    # Check required columns
    required_columns = [
        'Category', 'Price', 'Competitor Pricing', 'Discount', 
        'Weather Condition', 'Holiday/Promotion', 'Seasonality'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Convert categorical columns to dummy variables
    df_processed = pd.get_dummies(
        df, 
        columns=['Weather Condition', 'Seasonality', 'Region'],
        drop_first=True
    )
    
    # Get unique product categories
    categories = df['Category'].unique()
    print(f"Data loaded. Found {len(categories)} product categories: {', '.join(categories)}")
    
    # Models dictionary to store fitted models for each category
    models = {}
    
    # Fit models for each category
    print("Fitting competitor pricing models for each category...")
    for category in categories:
        print(f"Fitting model for {category}...")
        
        # Filter data for this category
        category_data = df_processed[df_processed['Category'] == category].copy()
        
        # Analyze competitor pricing strategy
        # Calculate price ratio (competitor price / our price)
        category_data['price_ratio'] = category_data['Competitor Pricing'] / category_data['Price']
        
        # Calculate statistics
        price_ratio_mean = category_data['price_ratio'].mean()
        price_ratio_std = category_data['price_ratio'].std()
        price_ratio_min = category_data['price_ratio'].min()
        price_ratio_max = category_data['price_ratio'].max()
        
        # Calculate percentage of times competitor price is lower, higher, or similar
        price_lower = (category_data['Competitor Pricing'] < category_data['Price']).mean() * 100
        price_higher = (category_data['Competitor Pricing'] > category_data['Price']).mean() * 100
        price_similar = (abs(category_data['Competitor Pricing'] - category_data['Price']) 
                         < 0.05 * category_data['Price']).mean() * 100
        
        # Check for correlation between prices
        price_correlation = category_data[['Price', 'Competitor Pricing']].corr().iloc[0, 1]
        
        # Prepare features
        X_columns = [col for col in category_data.columns if col not in 
                    ['Category', 'Competitor Pricing', 'price_ratio', 'Date', 'Store ID', 'Product ID']]
        
        X = category_data[X_columns]
        y = category_data['Competitor Pricing']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        if use_advanced_model:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (for Random Forest)
        if use_advanced_model:
            feature_importance = dict(zip(X_columns, model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:3]
        else:
            feature_importance = dict(zip(X_columns, model.coef_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:3]
        
        # Store model and insights
        models[category] = {
            'model': model,
            'scaler': scaler,
            'feature_names': X_columns,
            'price_ratio_mean': price_ratio_mean,
            'price_ratio_std': price_ratio_std,
            'price_ratio_min': price_ratio_min,
            'price_ratio_max': price_ratio_max,
            'price_lower_pct': price_lower,
            'price_higher_pct': price_higher,
            'price_similar_pct': price_similar,
            'price_correlation': price_correlation,
            'mse': mse,
            'r2': r2,
            'top_features': top_features
        }
        
        # Print insights
        print(f"Competitor pricing insights for {category}:")
        print(f"  Price ratio (competitor/our): {price_ratio_mean:.2f} ± {price_ratio_std:.2f}")
        print(f"  Competitor price is lower: {price_lower:.1f}% of the time")
        print(f"  Competitor price is higher: {price_higher:.1f}% of the time")
        print(f"  Competitor price is similar: {price_similar:.1f}% of the time")
        print(f"  Price correlation: {price_correlation:.2f}")
        print(f"  Model performance: MSE={mse:.2f}, R²={r2:.2f}")
        print(f"  Top influential features: {[f[0] for f in top_features]}")
        print("-" * 50)
    
    # Save models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"Models saved to {save_path}")
    return models

def predict_competitor_price(
        our_price: float,
        discount: float,
        is_promotion: bool,
        weather: str,
        season: str,
        region: str,
        category: str,
        model_path: str = 'models/saved/competitor_models.pkl'
) -> float:
    """
    Predict competitor price based on our price and other factors.
    
    Args:
        our_price: Our product price
        discount: Discount percentage
        is_promotion: Whether it's a promotion period
        weather: Weather condition
        season: Season
        region: Region
        category: Product category
        model_path: Path to the saved models
        
    Returns:
        Predicted competitor price
    """
    # Load models
    try:
        # First check in models/saved directory
        model_paths = [
            'models/saved/competitor_models.pkl',
            'competitor_models.pkl',
            os.path.join(os.path.dirname(__file__), '..', 'competitor_models.pkl')
        ]
        
        models = None
        for path in model_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models = pickle.load(f)
                break
        
        if models is None:
            raise FileNotFoundError(f"Competitor model file not found in expected locations")
    except FileNotFoundError:
        raise FileNotFoundError(f"Competitor model file not found: {model_path}")
    except Exception as e:
        raise Exception(f"Error loading competitor model: {e}")
    
    # Check if we have a model for this category
    if category not in models:
        raise ValueError(f"No competitor model found for category '{category}'")
    
    # Get model and related data
    model_data = models[category]
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Fallback to simple heuristic if confidence is low
    if model_data['r2'] < 0.5:
        # Use price ratio stats for a more reliable estimate
        ratio = model_data['price_ratio_mean']
        # Add some noise based on standard deviation
        ratio_noise = np.random.normal(0, model_data['price_ratio_std'] / 2)
        competitor_price = our_price * (ratio + ratio_noise)
        # Ensure price is within reasonable bounds
        min_ratio = max(0.5, model_data['price_ratio_min'])
        max_ratio = min(1.5, model_data['price_ratio_max'])
        competitor_price = max(our_price * min_ratio, min(our_price * max_ratio, competitor_price))
        return float(competitor_price)
    
    # Create feature vector
    # This requires careful matching of features with model's expected features
    # Create dummy variables for categorical features
    features = {
        'Price': our_price,
        'Discount': discount,
        'Holiday/Promotion': 1 if is_promotion else 0,
        'Inventory Level': 100,  # Default value
        'Demand Forecast': 80,   # Default value
    }
    
    # Add weather, season, and region dummies
    weather_cols = [col for col in feature_names if col.startswith('Weather Condition_')]
    for col in weather_cols:
        condition = col.replace('Weather Condition_', '')
        features[col] = 1 if condition == weather else 0
    
    season_cols = [col for col in feature_names if col.startswith('Seasonality_')]
    for col in season_cols:
        s = col.replace('Seasonality_', '')
        features[col] = 1 if s == season else 0
        
    region_cols = [col for col in feature_names if col.startswith('Region_')]
    for col in region_cols:
        r = col.replace('Region_', '')
        features[col] = 1 if r == region else 0
    
    # Create dataframe with all required features in correct order
    X = pd.DataFrame([features])
    X = X.reindex(columns=feature_names, fill_value=0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    competitor_price = model.predict(X_scaled)[0]
    
    return float(competitor_price)

if __name__ == "__main__":
    # Fit and save models
    models = fit_competitor_price_models()
    
    # Test predictions
    categories = list(models.keys())
    for category in categories:
        price = 50.0
        pred_price = predict_competitor_price(
            our_price=price,
            discount=10,
            is_promotion=True,
            weather="Sunny",
            season="Summer",
            region="North",
            category=category
        )
        price_ratio = pred_price / price
        print(f"{category}: Our price=${price:.2f}, Predicted competitor price=${pred_price:.2f} " +
              f"(ratio: {price_ratio:.2f})") 