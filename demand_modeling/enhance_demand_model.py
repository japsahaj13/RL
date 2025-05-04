import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pickle
import math
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import argparse

DATA_PATH = 'data/retail_store_inventory.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced demand model fitting')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                      help='Path to the retail store inventory CSV file')
    return parser.parse_args()

def load_and_clean_data(path=DATA_PATH):
    """Load, validate and clean the retail data"""
    print(f"Loading data from {path}...")
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"Warning: File {path} not found. Checking alternative paths...")
        alternative_paths = [
            'retail_store_inventory.csv',
            '../data/retail_store_inventory.csv',
            os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found data at {alt_path}")
                path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find retail_store_inventory.csv in any expected location")
            
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Warning: Found {missing_values.sum()} missing values")
        print(missing_values[missing_values > 0])
        # Handle missing values
        df = df.dropna(subset=['Units Sold', 'Price', 'Competitor Pricing'])
    
    # Check for zeros and negatives in key columns
    for col in ['Units Sold', 'Price', 'Competitor Pricing']:
        zeros = (df[col] == 0).sum()
        negatives = (df[col] < 0).sum()
        if zeros > 0 or negatives > 0:
            print(f"Warning: {col} has {zeros} zeros and {negatives} negatives")
    
    # Handle zeros in Units Sold - replace with small value to allow log transform
    if (df['Units Sold'] == 0).sum() > 0:
        min_nonzero = df[df['Units Sold'] > 0]['Units Sold'].min()
        small_value = min(1.0, min_nonzero / 10)
        print(f"Replacing {(df['Units Sold'] == 0).sum()} zeros in Units Sold with {small_value}")
        df.loc[df['Units Sold'] == 0, 'Units Sold'] = small_value
    
    # Identify and handle outliers in Units Sold
    q1 = df['Units Sold'].quantile(0.25)
    q3 = df['Units Sold'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers = (df['Units Sold'] > upper_bound).sum()
    if outliers > 0:
        print(f"Warning: Found {outliers} outliers in Units Sold (> {upper_bound:.2f})")
        cap_value = df['Units Sold'].quantile(0.95)
        print(f"Capping outliers to 95th percentile: {cap_value:.2f}")
        df.loc[df['Units Sold'] > cap_value, 'Units Sold'] = cap_value
    
    # Convert categorical variables to dummy variables
    df['Promotion'] = df['Holiday/Promotion'].astype(int)
    
    # Encode seasonality
    season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    df['SeasonValue'] = df['Seasonality'].map(season_map)
    
    # Create season dummy variables
    for season in season_map.keys():
        df[f'Season_{season}'] = (df['Seasonality'] == season).astype(int)
    
    # Encode weather
    weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2, 'Snowy': 3}
    df['WeatherValue'] = df['Weather Condition'].map(lambda x: weather_map.get(x, 0))
    
    # Create weather dummy variables
    for weather in weather_map.keys():
        df[f'Weather_{weather}'] = (df['Weather Condition'] == weather).astype(int)
    
    # Convert Region to numeric values
    regions = df['Region'].unique()
    region_map = {region: i for i, region in enumerate(regions)}
    df['RegionValue'] = df['Region'].map(region_map)
    
    # Create region dummy variables
    for region in regions:
        df[f'Region_{region}'] = (df['Region'] == region).astype(int)
    
    # Apply log transformation
    df['log_units_sold'] = np.log(df['Units Sold'])
    df['log_price'] = np.log(df['Price'] + 0.01)
    df['log_competitor_price'] = np.log(df['Competitor Pricing'] + 0.01)
    
    # Feature engineering: Add interaction terms
    df['price_ratio'] = df['Price'] / df['Competitor Pricing']
    df['log_price_ratio'] = df['log_price'] - df['log_competitor_price']
    
    # Price difference from competitor
    df['price_diff'] = df['Price'] - df['Competitor Pricing']
    df['price_discount'] = (df['Competitor Pricing'] - df['Price']) / df['Competitor Pricing']
    
    # Interaction with promotion
    df['promo_price_interaction'] = df['Promotion'] * df['log_price']
    
    # Weekly patterns (if date column exists)
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        for day in range(7):
            df[f'Day_{day}'] = (df['DayOfWeek'] == day).astype(int)
    except:
        print("Warning: Could not parse Date column for day-of-week feature")
    
    print("Data validation and cleaning complete.")
    return df

def fit_enhanced_models(df):
    """Fit both basic and enhanced regression models for each category"""
    print("Fitting demand models for each category...")
    
    categories = df['Category'].unique()
    models = {}
    params = {}
    
    results = []
    
    for category in categories:
        print(f"Fitting model for {category}...")
        category_data = df[df['Category'] == category]
        
        # Prepare basic features (compatible with simulator)
        X_basic = category_data[['log_price', 'log_competitor_price', 'Promotion', 
                               'SeasonValue', 'WeatherValue', 'RegionValue']]
        
        # Prepare enhanced features
        X_enhanced = category_data[[
            'log_price', 'log_competitor_price', 'Promotion',
            'price_ratio', 'log_price_ratio', 'price_discount',
            'promo_price_interaction'
        ]]
        
        # Add dummy variables for season and weather
        for col in category_data.columns:
            if col.startswith(('Season_', 'Weather_', 'Region_', 'Day_')):
                X_enhanced[col] = category_data[col]
        
        y = category_data['log_units_sold']
        
        # First fit basic model (compatible with simulator)
        basic_model = LinearRegression()
        basic_model.fit(X_basic, y)
        
        # Evaluate basic model
        y_pred_basic = basic_model.predict(X_basic)
        r2_basic = r2_score(y, y_pred_basic)
        mse_basic = mean_squared_error(y, y_pred_basic)
        
        # Then fit enhanced model
        enhanced_model = LinearRegression()
        enhanced_model.fit(X_enhanced, y)
        
        # Evaluate enhanced model
        y_pred_enhanced = enhanced_model.predict(X_enhanced)
        r2_enhanced = r2_score(y, y_pred_enhanced)
        mse_enhanced = mean_squared_error(y, y_pred_enhanced)
        
        # Extract parameters from basic model (for simulator compatibility)
        alpha = basic_model.intercept_  # log base demand
        price_elasticity = basic_model.coef_[0]  # should be negative
        competitor_sensitivity = basic_model.coef_[1]  # should be positive
        promotion_effect = basic_model.coef_[2]
        season_effect = basic_model.coef_[3]
        weather_effect = basic_model.coef_[4]
        region_effect = basic_model.coef_[5]
        
        # Add to results for comparison
        results.append({
            'Category': category,
            'R2_Basic': r2_basic,
            'R2_Enhanced': r2_enhanced,
            'MSE_Basic': mse_basic,
            'MSE_Enhanced': mse_enhanced,
            'Improvement': (r2_enhanced - r2_basic) / max(0.001, r2_basic) * 100
        })
        
        # Store the models
        models[category] = {
            'basic': basic_model,
            'enhanced': enhanced_model,
            'basic_features': list(X_basic.columns),
            'enhanced_features': list(X_enhanced.columns)
        }
        
        # Store parameters (same structure as before for simulator compatibility)
        params[category] = {
            'base_demand': math.exp(alpha),
            'price_elasticity': price_elasticity,
            'competitor_sensitivity': competitor_sensitivity,
            'promotion_multiplier': math.exp(promotion_effect),
            'season_effects': {
                'Spring': math.exp(0 * season_effect),
                'Summer': math.exp(1 * season_effect),
                'Autumn': math.exp(2 * season_effect),
                'Winter': math.exp(3 * season_effect)
            },
            'weather_effects': {
                'Sunny': math.exp(0 * weather_effect),
                'Cloudy': math.exp(1 * weather_effect),
                'Rainy': math.exp(2 * weather_effect),
                'Snowy': math.exp(3 * weather_effect)
            },
            'region_effects': {}
        }
        
        # Get list of regions from the dataset
        regions = df['Region'].unique()
        for i, region in enumerate(regions):
            params[category]['region_effects'][region] = math.exp(i * region_effect)
        
        print(f"Model fitted for {category}.")
        print(f"Base demand: {params[category]['base_demand']:.2f}")
        print(f"Price elasticity: {params[category]['price_elasticity']:.2f}")
        print(f"Competitor sensitivity: {params[category]['competitor_sensitivity']:.2f}")
        print(f"Promotion multiplier: {params[category]['promotion_multiplier']:.2f}")
        print(f"Basic model: R²: {r2_basic:.3f}, MSE: {mse_basic:.3f}")
        print(f"Enhanced model: R²: {r2_enhanced:.3f}, MSE: {mse_enhanced:.3f}")
        print(f"Improvement: {((r2_enhanced - r2_basic) / max(0.001, r2_basic) * 100):.1f}% in R²")
        print("--------------------------------------------------")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nComparison of Basic vs Enhanced Models:")
    print(results_df)
    
    # Create plot comparing R² values
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(categories))
    
    plt.bar([i for i in index], results_df['R2_Basic'], bar_width, label='Basic Model', color='blue', alpha=0.7)
    plt.bar([i + bar_width for i in index], results_df['R2_Enhanced'], bar_width, label='Enhanced Model', color='green', alpha=0.7)
    
    plt.xlabel('Category')
    plt.ylabel('R²')
    plt.title('Comparison of Model Performance (R²)')
    plt.xticks([i + bar_width/2 for i in index], categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig('demand_model_comparison.png')
    print("Saved model comparison chart to demand_model_comparison.png")
    
    return models, params, results_df

def save_models(params, models, results):
    """Save the fitted models and parameters"""
    # Ensure models/saved directory exists
    os.makedirs('models/saved', exist_ok=True)
    
    # Save standard parameters for simulator
    with open('models/saved/demand_models.pkl', 'wb') as f:
        pickle.dump(params, f)
    print(f"Compatible demand models saved to models/saved/demand_models.pkl")
    
    # Save enhanced models for reference
    with open('models/saved/enhanced_demand_models.pkl', 'wb') as f:
        pickle.dump({
            'params': params,
            'models': models,
            'results': results
        }, f)
    print(f"Enhanced models saved to models/saved/enhanced_demand_models.pkl")

def main():
    """Run the enhanced demand model fitting process"""
    args = parse_args()
    df = load_and_clean_data(args.data)
    models, params, results = fit_enhanced_models(df)
    save_models(params, models, results)
    
    # Print parameters for MSME environment to verify format
    print("\nVerifying parameter format for MSME simulator:")
    print("Electronics parameters:")
    print(f"Base demand: {params['Electronics']['base_demand']}")
    print(f"Price elasticity: {params['Electronics']['price_elasticity']}")
    print(f"Competitor sensitivity: {params['Electronics']['competitor_sensitivity']}")
    print(f"Promotion multiplier: {params['Electronics']['promotion_multiplier']}")
    print(f"Season effects: {list(params['Electronics']['season_effects'].items())[:2]}...")
    print(f"Weather effects: {list(params['Electronics']['weather_effects'].items())[:2]}...")
    print(f"Region effects: {list(params['Electronics']['region_effects'].items())[:2]}...")

if __name__ == "__main__":
    main() 