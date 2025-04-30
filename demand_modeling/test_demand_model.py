import numpy as np
import matplotlib.pyplot as plt
from gym_working import MSMEConfig, MSMEEnvironment
import pickle
import pandas as pd
import os

def fit_demand_models_and_save():
    """
    Run the demand model fitting and save the parameters
    """
    try:
        from demand_model_fitting import fit_demand_models
        params = fit_demand_models()
        print("Demand models fitted and saved successfully!")
        return True
    except Exception as e:
        print(f"Error fitting demand models: {e}")
        return False

def test_demand_model_with_price_changes():
    """
    Test the demand model with different price points and visualize the results
    """
    # Check if demand_models.pkl exists
    if not os.path.exists('demand_models.pkl'):
        # Try to fit models from dataset
        print("No saved demand models found. Attempting to fit models from dataset...")
        fit_demand_models_and_save()
    
    # Create a config with fitted model
    config = MSMEConfig(
        product_name="Test Product",
        product_category="Electronics",  # Change this to match a category in your dataset
        region="North",
        unit_cost=50,
        base_demand=200, 
        use_fitted_model=True  # Use the fitted model
    )
    
    # Create the environment
    env = MSMEEnvironment(config)
    
    # Reset the environment
    env.reset()
    
    # Test the demand model with varying prices
    test_prices = np.linspace(10, 100, 50)
    demands_fitted = []
    demands_legacy = []
    
    # Set up competitor price and promotion flag for testing
    comp_price = 50
    promo_flag = False
    time_step = 0
    
    for price in test_prices:
        # Save use_fitted_model setting
        original_setting = config.use_fitted_model
        
        # Test with fitted model
        config.use_fitted_model = True
        demand_fitted = env._demand_model(price, comp_price, promo_flag, time_step)
        demands_fitted.append(demand_fitted)
        
        # Test with legacy model
        config.use_fitted_model = False
        demand_legacy = env._demand_model(price, comp_price, promo_flag, time_step)
        demands_legacy.append(demand_legacy)
        
        # Restore original setting
        config.use_fitted_model = original_setting
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(test_prices, demands_fitted, label='Fitted Log-Log Model', color='blue')
    plt.plot(test_prices, demands_legacy, label='Legacy Linear Model', color='red', linestyle='--')
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.title('Demand Response to Price Changes')
    plt.legend()
    plt.grid(True)
    
    # Test with varying competitor prices
    test_comp_prices = np.linspace(10, 100, 50)
    demands_comp_fitted = []
    demands_comp_legacy = []
    
    # Fix our price for this test
    price = 50
    
    for comp_price in test_comp_prices:
        # Save use_fitted_model setting
        original_setting = config.use_fitted_model
        
        # Test with fitted model
        config.use_fitted_model = True
        demand_fitted = env._demand_model(price, comp_price, promo_flag, time_step)
        demands_comp_fitted.append(demand_fitted)
        
        # Test with legacy model
        config.use_fitted_model = False
        demand_legacy = env._demand_model(price, comp_price, promo_flag, time_step)
        demands_comp_legacy.append(demand_legacy)
        
        # Restore original setting
        config.use_fitted_model = original_setting
    
    # Plot competitor price response
    plt.subplot(2, 1, 2)
    plt.plot(test_comp_prices, demands_comp_fitted, label='Fitted Log-Log Model', color='blue')
    plt.plot(test_comp_prices, demands_comp_legacy, label='Legacy Linear Model', color='red', linestyle='--')
    plt.xlabel('Competitor Price')
    plt.ylabel('Demand')
    plt.title('Demand Response to Competitor Price Changes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demand_model_comparison.png')
    plt.show()

def test_full_market_simulation():
    """
    Run a full market simulation with the updated demand model
    """
    # Create a config with fitted model
    config = MSMEConfig(
        product_name="Market Test Product",
        product_category="Electronics",
        region="North",
        unit_cost=50,
        base_demand=200,
        use_fitted_model=True  # Use the fitted model
    )
    
    # Create the environment
    env = MSMEEnvironment(config, time_horizon=90)  # Run for 90 days
    
    # Reset the environment
    state = env.reset()[0]
    
    # Data collection
    prices = []
    demands = []
    sales = []
    inventories = []
    profits = []
    comp_prices = []
    
    # Simulate with a simple pricing strategy (for testing)
    done = False
    total_reward = 0
    
    while not done:
        # Simple price selection strategy
        if env.time_step % 30 < 10:
            # Undercut competitor slightly
            action = 3  # Select the pricing tier
        elif env.time_step % 30 < 20:
            # Match competitor
            action = 5
        else:
            # Premium pricing
            action = 7
        
        # Take a step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect data
        prices.append(info['price'])
        comp_prices.append(info['comp_price'])
        demands.append(info['demand'])
        sales.append(info['sales'])
        inventories.append(info['inventory'])
        profits.append(info['profit'])
        
        total_reward += reward
        state = next_state
    
    # Convert to numpy arrays
    prices = np.array(prices)
    comp_prices = np.array(comp_prices)
    demands = np.array(demands)
    sales = np.array(sales)
    inventories = np.array(inventories)
    profits = np.array(profits)
    
    # Plot the simulation results
    plt.figure(figsize=(15, 15))
    
    # Price and competitor price over time
    plt.subplot(3, 2, 1)
    plt.plot(prices, label='Our Price', color='blue')
    plt.plot(comp_prices, label='Competitor Price', color='red', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    plt.title('Price Dynamics')
    plt.legend()
    plt.grid(True)
    
    # Demand and sales over time
    plt.subplot(3, 2, 2)
    plt.plot(demands, label='Demand', color='green')
    plt.plot(sales, label='Sales', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Units')
    plt.title('Demand and Sales')
    plt.legend()
    plt.grid(True)
    
    # Inventory over time
    plt.subplot(3, 2, 3)
    plt.plot(inventories, label='Inventory', color='purple')
    plt.axhline(y=config.restock_level, color='red', linestyle='--', label='Restock Level')
    plt.xlabel('Time Step')
    plt.ylabel('Units')
    plt.title('Inventory Dynamics')
    plt.legend()
    plt.grid(True)
    
    # Profit over time
    plt.subplot(3, 2, 4)
    plt.plot(profits, label='Profit', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Profit ($)')
    plt.title('Profit Dynamics')
    plt.legend()
    plt.grid(True)
    
    # Price elasticity visualization
    plt.subplot(3, 2, 5)
    # Scatter plot of price vs demand
    plt.scatter(prices, demands, alpha=0.7, label='Price-Demand Points')
    # Fit curve through the points
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(prices), np.log(demands))
        x_range = np.linspace(min(prices), max(prices), 100)
        y_range = np.exp(intercept) * x_range**slope
        plt.plot(x_range, y_range, 'r--', label=f'Elasticity: {slope:.2f}')
    except:
        pass
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.title('Price Elasticity Visualization')
    plt.legend()
    plt.grid(True)
    
    # Cumulative profit
    plt.subplot(3, 2, 6)
    plt.plot(np.cumsum(profits), label='Cumulative Profit', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Profit ($)')
    plt.title('Cumulative Profit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('market_simulation.png')
    plt.show()
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total profit: {np.sum(profits):.2f}")
    print(f"Average price: {np.mean(prices):.2f}")
    print(f"Average competitor price: {np.mean(comp_prices):.2f}")
    print(f"Average demand: {np.mean(demands):.2f}")
    print(f"Average sales: {np.mean(sales):.2f}")
    print(f"Final inventory: {inventories[-1]:.2f}")

if __name__ == "__main__":
    print("Testing demand model and visualizing results...")
    test_demand_model_with_price_changes()
    test_full_market_simulation()
    print("Testing complete!") 