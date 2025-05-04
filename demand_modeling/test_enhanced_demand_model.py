import numpy as np
import matplotlib.pyplot as plt
from environments.config import MSMEConfig
from environments.msme_env import MSMEEnvironment
import pickle
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test enhanced demand model with MSME environment')
    parser.add_argument('--model', type=str, default='demand_models.pkl',
                      help='Path to the demand model pickle file')
    return parser.parse_args()

def load_demand_models(model_path='models/saved/demand_models.pkl'):
    """Load demand models from pickle file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please run enhance_demand_model.py first.")
    
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"Loaded demand models from {model_path}")
    
    # If it's the enhanced model format, extract just the params
    if isinstance(models, dict) and 'params' in models:
        print("Detected enhanced model format")
        params = models['params']
        results = models.get('results', None)
        if results is not None:
            print("\nModel performance:")
            print(results[['Category', 'R2_Basic', 'MSE_Basic']])
        return params
    
    return models

def test_demand_model_for_category(category, params):
    """Test the demand model for a specific product category"""
    print(f"\nTesting demand model for {category}...")
    
    # Create config with the fitted model parameters
    config = MSMEConfig(
        product_name=f"Test {category} Product",
        product_category=category,
        region="North",
        unit_cost=50,
        base_demand=params[category]['base_demand'],
        price_elasticity=params[category]['price_elasticity'],
        competitor_sensitivity=params[category]['competitor_sensitivity'],
        promotion_multiplier=params[category]['promotion_multiplier'],
        season_effects=params[category]['season_effects'],
        weather_effects=params[category]['weather_effects'],
        region_effects=params[category]['region_effects'],
        use_fitted_model=True
    )
    
    # Create environment
    env = MSMEEnvironment(config)
    
    # Reset environment
    env.reset()
    
    # Test demand model with varying prices
    test_prices = np.linspace(config.unit_cost * 0.8, config.unit_cost * 3, 20)
    comp_price = config.unit_cost * 1.5  # Fixed competitor price
    demands = []
    
    for price in test_prices:
        demand = env._demand_model(price, comp_price, False, 0)
        demands.append(demand)
    
    # Calculate price elasticity from the data
    log_prices = np.log(test_prices)
    log_demands = np.log(demands)
    
    # Simple regression to estimate elasticity
    slope = np.polyfit(log_prices, log_demands, 1)[0]
    
    print(f"Parameters for {category}:")
    print(f"  Base demand: {params[category]['base_demand']:.2f}")
    print(f"  Price elasticity: {params[category]['price_elasticity']:.2f}")
    print(f"  Empirical elasticity (from test): {slope:.2f}")
    print(f"  Competitor sensitivity: {params[category]['competitor_sensitivity']:.2f}")
    print(f"  Promotion multiplier: {params[category]['promotion_multiplier']:.2f}")
    
    return test_prices, demands, slope

def test_demand_response_to_comp_price(category, params):
    """Test how demand responds to competitor pricing"""
    config = MSMEConfig(
        product_name=f"Test {category} Product",
        product_category=category,
        region="North",
        unit_cost=50,
        use_fitted_model=True
    )
    
    env = MSMEEnvironment(config)
    env.reset()
    
    # Fixed our price
    our_price = config.unit_cost * 1.5
    
    # Test with varying competitor prices
    comp_prices = np.linspace(our_price * 0.7, our_price * 1.3, 20)
    demands = []
    
    for comp_price in comp_prices:
        demand = env._demand_model(our_price, comp_price, False, 0)
        demands.append(demand)
    
    # Calculate competitor sensitivity from the data
    log_comp_prices = np.log(comp_prices)
    log_demands = np.log(demands)
    
    # Simple regression to estimate competitor sensitivity
    slope = np.polyfit(log_comp_prices, log_demands, 1)[0]
    
    print(f"Competitor price test for {category}:")
    print(f"  Competitor sensitivity (model): {params[category]['competitor_sensitivity']:.2f}")
    print(f"  Empirical sensitivity (from test): {slope:.2f}")
    
    return comp_prices, demands, slope

def run_price_simulation(category, params):
    """Run a simple price simulation to test the environment"""
    config = MSMEConfig(
        product_name=f"Test {category} Product",
        product_category=category,
        region="North",
        unit_cost=50,
        use_fitted_model=True
    )
    
    # Set time horizon but create the environment directly with it
    time_horizon = 60
    env = MSMEEnvironment(config, time_horizon=time_horizon)
    
    state = env.reset()[0]
    
    # Track metrics
    days = []
    our_prices = []
    comp_prices = []
    demands = []
    sales = []
    inventories = []
    profits = []
    
    # Simulate with a simple pricing strategy
    for day in range(time_horizon):
        # Simple price strategy - cycle through different price levels
        if day % 15 < 5:
            # Low price
            price_tier = 2
        elif day % 15 < 10:
            # Medium price
            price_tier = 5
        else:
            # High price
            price_tier = 8
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(price_tier)
        
        # Record metrics
        days.append(day)
        our_prices.append(info['price'])
        comp_prices.append(info['comp_price'])
        demands.append(info['demand'])
        sales.append(info['sales'])
        inventories.append(info['inventory'])
        profits.append(info['profit'])
        
        state = next_state
        
        if terminated or truncated:
            break
    
    # Calculate cumulative metrics
    total_profit = sum(profits)
    total_sales = sum(sales)
    avg_price = sum(our_prices) / len(our_prices)
    avg_comp_price = sum(comp_prices) / len(comp_prices)
    
    print(f"\nSimulation results for {category}:")
    print(f"  Total profit: ${total_profit:.2f}")
    print(f"  Total sales: {total_sales:.0f} units")
    print(f"  Average price: ${avg_price:.2f}")
    print(f"  Average competitor price: ${avg_comp_price:.2f}")
    
    return {
        'days': days,
        'our_prices': our_prices,
        'comp_prices': comp_prices,
        'demands': demands,
        'sales': sales,
        'inventories': inventories,
        'profits': profits,
        'total_profit': total_profit
    }

def main():
    try:
        args = parse_args()
        
        # Load demand models
        params = load_demand_models(args.model)
        
        # Get all available categories
        categories = list(params.keys())
        print(f"Available categories: {', '.join(categories)}")
        
        # Set up figure
        plt.figure(figsize=(15, 12))
        
        # Test each category
        price_elasticity_results = {}
        competitor_sensitivity_results = {}
        simulation_results = {}
        
        for i, category in enumerate(categories):
            try:
                # Test price elasticity
                test_prices, demands, elasticity = test_demand_model_for_category(category, params)
                price_elasticity_results[category] = {
                    'prices': test_prices,
                    'demands': demands,
                    'elasticity': elasticity
                }
                
                # Test competitor sensitivity
                comp_prices, comp_demands, sensitivity = test_demand_response_to_comp_price(category, params)
                competitor_sensitivity_results[category] = {
                    'comp_prices': comp_prices,
                    'demands': comp_demands,
                    'sensitivity': sensitivity
                }
                
                # Run simulation
                sim_results = run_price_simulation(category, params)
                simulation_results[category] = sim_results
            except Exception as e:
                print(f"Error testing {category}: {e}")
        
        # Plot results if we have data
        if price_elasticity_results:
            # Plot price elasticity
            plt.subplot(3, 1, 1)
            for category in price_elasticity_results:
                results = price_elasticity_results[category]
                plt.plot(results['prices'], results['demands'], marker='o', label=f"{category} (e={results['elasticity']:.2f})")
            
            plt.title('Price Elasticity of Demand')
            plt.xlabel('Price ($)')
            plt.ylabel('Demand (units)')
            plt.grid(True)
            plt.legend()
            
            # Plot competitor sensitivity 
            plt.subplot(3, 1, 2)
            for category in competitor_sensitivity_results:
                results = competitor_sensitivity_results[category]
                plt.plot(results['comp_prices'], results['demands'], marker='o', label=f"{category} (s={results['sensitivity']:.2f})")
            
            plt.title('Demand Response to Competitor Pricing')
            plt.xlabel('Competitor Price ($)')
            plt.ylabel('Demand (units)')
            plt.grid(True)
            plt.legend()
            
            # Plot profits from simulation
            if simulation_results:
                plt.subplot(3, 1, 3)
                categories_with_sim = list(simulation_results.keys())
                bar_positions = range(len(categories_with_sim))
                profits = [simulation_results[cat]['total_profit'] for cat in categories_with_sim]
                plt.bar(bar_positions, profits)
                plt.xticks(bar_positions, categories_with_sim)
                plt.title('Simulation Results: Total Profit by Category')
                plt.xlabel('Product Category')
                plt.ylabel('Total Profit ($)')
            
            plt.tight_layout()
            plt.savefig('demand_model_test_results.png')
            print("\nSaved test results plot to demand_model_test_results.png")
        else:
            print("No results to plot. Check for errors above.")
    
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 