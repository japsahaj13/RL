#!/usr/bin/env python
"""
Script to generate competitor price models from the retail dataset.
This will analyze how competitors price their products relative to ours
and create models to predict competitor behavior.
"""

import os
from demand_modeling.competitor_price_modeling import fit_competitor_price_models

if __name__ == "__main__":
    print("Generating competitor price models...")
    
    # Create models/saved directory if it doesn't exist
    os.makedirs('models/saved', exist_ok=True)
    
    # Fit and save models
    models = fit_competitor_price_models(
        csv_path='data/retail_store_inventory.csv',
        save_path='models/saved/competitor_models.pkl',
        use_advanced_model=True  # Use RandomForest for better accuracy
    )
    
    print("\nCompetitor pricing model generation complete!")
    print("The models have been saved to models/saved/competitor_models.pkl")
    
    # Print summary
    print("\nSummary of competitor pricing strategies:")
    for category, model_data in models.items():
        print(f"\n{category}:")
        print(f"  Price ratio (competitor/our): {model_data['price_ratio_mean']:.2f} ± {model_data['price_ratio_std']:.2f}")
        print(f"  Competitor price is lower: {model_data['price_lower_pct']:.1f}% of the time")
        print(f"  Competitor price is higher: {model_data['price_higher_pct']:.1f}% of the time")
        print(f"  Price correlation: {model_data['price_correlation']:.2f}")
        print(f"  Model performance (R²): {model_data['r2']:.2f}")
        print(f"  Top influential features: {[f[0] for f in model_data['top_features']]}")
    
    print("\nYou can now use these models to simulate realistic competitor pricing behavior.")
    print("The msme_env.py has been updated to use these models when available.") 