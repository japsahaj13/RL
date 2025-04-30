import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pickle
import math

class DemandModelFitter:
    def __init__(self, data_path='data/retail_store_inventory.csv'):
        self.data_path = data_path
        self.models = {}
        self.params = {}
        
    def load_data(self):
        """Load and preprocess the retail data"""
        print(f"Loading data from {self.data_path}...")
        # Check if file exists
        if not os.path.exists(self.data_path):
            print(f"Warning: File {self.data_path} not found. Checking alternative paths...")
            alternative_paths = [
                'retail_store_inventory.csv',
                '../data/retail_store_inventory.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_store_inventory.csv')
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found data at {alt_path}")
                    self.data_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find retail_store_inventory.csv in any expected location")
                
        self.df = pd.read_csv(self.data_path)
        
        # Clean data - handle missing values
        self.df = self.df.dropna(subset=['Units Sold', 'Price', 'Competitor Pricing'])
        
        # Convert categorical variables to dummy variables
        self.df['Promotion'] = self.df['Holiday/Promotion'].astype(int)
        
        # Encode seasonality as numeric values
        season_map = {
            'Spring': 0, 
            'Summer': 1, 
            'Autumn': 2, 
            'Winter': 3
        }
        self.df['SeasonValue'] = self.df['Seasonality'].map(season_map)
        
        # Encode weather as numeric values
        weather_map = {
            'Sunny': 0,
            'Cloudy': 1,
            'Rainy': 2,
            'Snowy': 3
        }
        self.df['WeatherValue'] = self.df['Weather Condition'].map(lambda x: 
                                                                   weather_map.get(x, 0))
        
        # Convert Region to numeric values
        regions = self.df['Region'].unique()
        region_map = {region: i for i, region in enumerate(regions)}
        self.df['RegionValue'] = self.df['Region'].map(region_map)
        
        # Apply log transformation (adding small constant to avoid log(0))
        self.df['log_units_sold'] = np.log(self.df['Units Sold'] + 1)
        self.df['log_price'] = np.log(self.df['Price'] + 0.01)
        self.df['log_competitor_price'] = np.log(self.df['Competitor Pricing'] + 0.01)
        
        # Get unique categories for modeling
        self.categories = self.df['Category'].unique()
        print(f"Data loaded. Found {len(self.categories)} product categories: {', '.join(self.categories)}")
        
    def fit_models(self):
        """Fit log-log regression models for each product category"""
        print("Fitting demand models for each category...")
        
        for category in self.categories:
            print(f"Fitting model for {category}...")
            category_data = self.df[self.df['Category'] == category]
            
            # Prepare features and target
            X = category_data[['log_price', 'log_competitor_price', 'Promotion', 
                               'SeasonValue', 'WeatherValue', 'RegionValue']]
            y = category_data['log_units_sold']
            
            # Fit the model
            model = LinearRegression()
            model.fit(X, y)
            
            # Extract parameters
            alpha = model.intercept_  # log base demand
            price_elasticity = model.coef_[0]  # should be negative
            competitor_sensitivity = model.coef_[1]  # should be positive
            promotion_effect = model.coef_[2]
            season_effect = model.coef_[3]
            weather_effect = model.coef_[4]
            region_effect = model.coef_[5]
            
            # Store the model
            self.models[category] = model
            
            # Store parameters
            self.params[category] = {
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
            regions = self.df['Region'].unique()
            for i, region in enumerate(regions):
                self.params[category]['region_effects'][region] = math.exp(i * region_effect)
            
            print(f"Model fitted for {category}.")
            print(f"Base demand: {self.params[category]['base_demand']:.2f}")
            print(f"Price elasticity: {self.params[category]['price_elasticity']:.2f}")
            print(f"Competitor sensitivity: {self.params[category]['competitor_sensitivity']:.2f}")
            print(f"Promotion multiplier: {self.params[category]['promotion_multiplier']:.2f}")
            print("--------------------------------------------------")
        
    def save_models(self, output_path='demand_models.pkl'):
        """Save the fitted models and parameters to a pickle file"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Models saved to {output_path}")
    
    def run(self, output_path='demand_models.pkl'):
        """Run the entire fitting process"""
        self.load_data()
        self.fit_models()
        self.save_models(output_path)

def fit_demand_models():
    """Helper function to fit demand models"""
    fitter = DemandModelFitter()
    fitter.run()
    return fitter.params

if __name__ == "__main__":
    fit_demand_models() 