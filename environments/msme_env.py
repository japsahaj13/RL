"""
MSME Pricing Environment for RL-based price optimization.

This module implements a Gym-compatible environment that simulates
a retail business with inventory management, competitor pricing,
and demand dynamics.
"""

import gym
import numpy as np
import random
import torch
import os
import math
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union

from environments.config import MSMEConfig

# Try to import competitor price modeling
try:
    from demand_modeling.competitor_price_modeling import predict_competitor_price
except ImportError:
    print("Warning: competitor_price_modeling module not found. Using simulated competitor behavior.")
    predict_competitor_price = None

class MSMEEnvironment(gym.Env):
    """
    Micro, Small & Medium Enterprise (MSME) pricing environment.
    
    This environment simulates a retail business where an agent needs to make
    pricing decisions based on inventory, competitor pricing, and market demand.
    """
    
    def __init__(
            self, config: MSMEConfig, time_horizon: int = 30, 
            alpha: float = 0.2, beta: float = 0.5, gamma: float = 0.2, delta: float = 0.2,
            use_data_driven_competitor: bool = True
    ):
        """
        Initialize the MSME pricing environment.
        
        Args:
            config: Configuration object with environment parameters
            time_horizon: Number of time steps for each episode
            alpha: Weight for revenue in reward function
            beta: Weight for market share in reward function
            gamma: Weight for inventory management in reward function
            delta: Weight for profit margin in reward function
            use_data_driven_competitor: Whether to use data-driven competitor pricing model
        """
        self.config = config
        self.time_horizon = time_horizon
        
        # Reward shaping weights
        self.alpha = alpha  # Revenue
        self.beta = beta    # Market share
        self.gamma = gamma  # Inventory management
        self.delta = delta  # Profit margin
        
        # Whether to use data-driven competitor pricing
        self.use_data_driven_competitor = use_data_driven_competitor and predict_competitor_price is not None
        
        if self.use_data_driven_competitor:
            print(f"Using data-driven competitor pricing model for {config.product_category}")
        else:
            if use_data_driven_competitor and predict_competitor_price is None:
                print("Data-driven competitor model not available. Using simulated competitor behavior.")
            print(f"Using simulated competitor behavior for {config.product_category}")
        
        # Define action space (discrete price tiers)
        self.action_space = spaces.Discrete(len(config.price_tiers))
        
        # Define observation space (6 continuous values)
        # [inventory, competitor_price, last_demand, last_sales, price, time_step]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([
                config.initial_inventory * 2,  # Inventory can grow with restocking
                config.unit_cost * 3,          # Max competitor price
                config.base_demand * 3,        # Max demand
                config.base_demand * 3,        # Max sales
                config.unit_cost * 3,          # Max price
                self.time_horizon              # Time step
            ]),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.inventory = 0
        self.comp_price = 0
        self.last_demand = 0
        self.last_sales = 0
        self.price = 0
        self.time_step = 0
        self.done = False
        
        # Additional tracking variables
        self.sales_history = []
        self.price_history = []
        self.demand_history = []
        self.inventory_history = []
        self.comp_price_history = []
        self.profit_history = []
        
        # Season and weather
        self.seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        self.current_season = random.choice(self.seasons)
        
        self.weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
        self.current_weather = random.choice(self.weather_conditions)
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional configuration parameters
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset state variables
        self.inventory = self.config.initial_inventory
        self.comp_price = self.config.unit_cost * (1 + 0.2 * np.random.randn())
        self.comp_price = max(self.comp_price, self.config.unit_cost * 0.5)
        self.last_demand = 0
        self.last_sales = 0
        self.price = self.config.price_tiers[len(self.config.price_tiers) // 2]  # Start with middle price tier
        self.time_step = 0
        self.done = False
        
        # Reset history
        self.sales_history = []
        self.price_history = []
        self.demand_history = []
        self.inventory_history = []
        self.comp_price_history = []
        self.profit_history = []
        
        # Reset restock scheduling
        if hasattr(self.config, 'get_restock_parameters'):
            # Get initial parameters
            initial_params = self.config.get_restock_parameters(
                current_time=0,
                demand_forecast=self.config.base_demand,
                price=self.price,
                units_sold=0
            )
            # Schedule first check
            self.next_restock_check_time = initial_params['restock_period']
        
        # Randomly initialize season and weather
        self.current_season = random.choice(self.seasons)
        self.current_weather = random.choice(self.weather_conditions)
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            Numpy array with the state representation
        """
        # Construct raw state
        state = np.array([
            self.inventory,
            self.comp_price,
            self.last_demand,
            self.last_sales,
            self.price,
            self.time_step
        ], dtype=np.float32)
        
        # Normalize state if needed
        # state = self._normalize_state(state)
        
        return state
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize the state values to the range [0, 1].
        
        Args:
            state: Raw state values
            
        Returns:
            Normalized state values
        """
        # Define normalization factors
        norm_factors = np.array([
            self.config.initial_inventory * 2,  # Inventory
            self.config.unit_cost * 3,          # Competitor price
            self.config.base_demand * 3,        # Last demand
            self.config.base_demand * 3,        # Last sales
            self.config.unit_cost * 3,          # Price
            self.time_horizon                   # Time step
        ], dtype=np.float32)
        
        # Normalize
        return state / norm_factors
    
    def _demand_model(self, price: float, comp_price: float, promo_flag: bool, time_step: int) -> float:
        """
        Calculate demand based on price, competitor price, and other factors.
        
        Args:
            price: Our price
            comp_price: Competitor's price
            promo_flag: Whether there's a promotion
            time_step: Current time step
            
        Returns:
            Calculated demand
        """
        # Log-log demand model based on fitted parameters from demand models
        # Base demand adjusted for season, weather, and region
        base_demand = self.config.base_demand
        seasonal_effect = self.config.season_effects.get(self.current_season, 1.0)
        weather_effect = self.config.weather_effects.get(self.current_weather, 1.0)
        region_effect = self.config.region_effects.get(self.config.region, 1.0)
        
        adjusted_base_demand = base_demand * seasonal_effect * weather_effect * region_effect
        
        # Apply promotion multiplier if applicable
        if promo_flag:
            adjusted_base_demand *= self.config.promotion_multiplier
        
        # Apply price elasticity and competitor sensitivity using log-log model
        # log(D) = log(D0) + e*log(P) + c*log(P_comp)
        # D = D0 * P^e * P_comp^c
        try:
            demand = adjusted_base_demand * (price ** self.config.price_elasticity) * (comp_price ** self.config.competitor_sensitivity)
            
            # Add random noise
            noise = 1.0 + np.random.normal(0, self.config.demand_noise_scale)
            demand *= max(0.5, noise)  # Ensure noise doesn't reduce demand too much
            
            return max(0, demand)
        except Exception as e:
            print(f"Error in log-log demand model: {e}")
            print("This is a critical error - check your demand model parameters!")
            # Return a reasonable fallback to prevent total failure
            return self.config.base_demand
    
    def _is_promotion_period(self) -> bool:
        """
        Check if the current time step is a promotion period.
        
        Returns:
            True if it's a promotion period, False otherwise
        """
        # Simple promotion logic: every 7th day
        return self.time_step % 7 == 0
    
    def _sample_competitor_price(self, our_price: float) -> float:
        """
        Sample the competitor's price for the current time step.
        
        Args:
            our_price: Our current price
            
        Returns:
            Competitor's price
        """
        # Try to use the data-driven competitor model if enabled
        if self.use_data_driven_competitor:
            try:
                # Get current state variables
                discount = 0  # Default discount (could calculate from price_tiers)
                is_promotion = self._is_promotion_period()
                
                # Get competitor price prediction from the model
                competitor_price = predict_competitor_price(
                    our_price=our_price,
                    discount=discount,
                    is_promotion=is_promotion,
                    weather=self.current_weather,
                    season=self.current_season,
                    region=self.config.region,
                    category=self.config.product_category
                )
                
                # Define valid price range for competitor
                min_price = self.config.unit_cost * 0.7
                max_price = self.config.unit_cost * 2.2
                
                # Ensure price is within valid range
                competitor_price = max(min_price, min(max_price, competitor_price))
                
                # Add small random noise for variation (1% max fluctuation)
                noise_factor = 1.0 + 0.01 * (2 * random.random() - 1)
                competitor_price *= noise_factor
                
                return competitor_price
                
            except Exception as e:
                print(f"Warning: Error using data-driven competitor pricing model: {e}")
                print("Falling back to simulated competitor behavior.")
                # Fall back to the simulation approach below
        
        # Simulation-based competitor behavior (original implementation)
        # Define valid price range for competitor
        min_price = self.config.unit_cost * 0.8
        max_price = self.config.unit_cost * 2.0
        
        # Competitor reacts to our price with some delay and randomness
        if len(self.price_history) > 0:
            # Competitor has a 10% chance of changing price each day
            if random.random() < 0.1:
                # React to our price from a few steps ago
                lookback = min(3, len(self.price_history))
                our_previous_price = self.price_history[-lookback]
                
                # Random strategy: sometimes undercut, sometimes go higher
                strategy = random.random()
                
                if strategy < 0.6:  # 60% chance to undercut
                    new_price = our_previous_price * (0.9 + 0.05 * random.random())
                elif strategy < 0.9:  # 30% chance to go higher
                    new_price = our_previous_price * (1.1 + 0.1 * random.random())
                else:  # 10% chance to match
                    new_price = our_previous_price * (0.98 + 0.04 * random.random())
                
                # Ensure price is within valid range
                self.comp_price = max(min_price, min(max_price, new_price))
        
        # Add some random walk to the competitor price
        random_factor = 1.0 + 0.02 * (random.random() - 0.5)
        self.comp_price *= random_factor
        
        # Ensure price is within valid range
        return max(min_price, min(max_price, self.comp_price))
    
    def _restock_inventory(self) -> Tuple[float, bool]:
        """
        Handle inventory restocking logic with dynamic scheduling.
        
        Returns:
            Tuple of (restocking cost, whether restock occurred)
        """
        # Store current inventory for prediction
        self.config._current_inventory = self.inventory
        
        # Get forecast and recent sales
        forecast_demand = self._get_demand_forecast()
        units_sold = self.last_sales if self.sales_history else 0
        
        # Initialize next_check_time if not set
        if not hasattr(self, 'next_restock_check_time'):
            # Get initial parameters
            params = self.config.get_restock_parameters(
                current_time=self.time_step,
                demand_forecast=forecast_demand,
                price=self.price,
                units_sold=units_sold
            )
            # Schedule first check
            self.next_restock_check_time = params['restock_period']
        
        # Check if it's time to review inventory
        if self.time_step >= self.next_restock_check_time:
            # Get current parameters
            params = self.config.get_restock_parameters(
                current_time=self.time_step,
                demand_forecast=forecast_demand,
                price=self.price,
                units_sold=units_sold
            )
            
            # Check if inventory is below restock level
            if self.inventory < params['restock_level']:
                # Add to inventory
                self.inventory += params['restock_amount']
                
                # Calculate cost
                restock_cost = params['restock_amount'] * self.config.unit_cost
                
                # Schedule next check
                self.next_restock_check_time = self.time_step + params['restock_period']
                
                return restock_cost, True
            else:
                # Still schedule next check even if no restock needed
                self.next_restock_check_time = self.time_step + params['restock_period']
        
        # No restocking
        return 0.0, False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Action index (representing a price tier)
            
        Returns:
            Tuple of (next state, reward, terminated, truncated, info)
        """
        # Check if episode is already done
        if self.done:
            next_state = self._get_state()
            return next_state, 0.0, True, False, {}
        
        # Set the price based on action
        self.price = self.config.price_tiers[action]
        self.price_history.append(self.price)
        
        # Update competitor price
        self.comp_price = self._sample_competitor_price(self.price)
        self.comp_price_history.append(self.comp_price)
        
        # Calculate demand based on price and other factors
        promo_flag = self._is_promotion_period()
        demand = self._demand_model(self.price, self.comp_price, promo_flag, self.time_step)
        self.last_demand = demand
        self.demand_history.append(demand)
        
        # Calculate sales (min of demand and inventory)
        sales = min(demand, self.inventory)
        self.last_sales = sales
        self.sales_history.append(sales)
        
        # Update inventory
        self.inventory -= sales
        self.inventory_history.append(self.inventory)
        
        # Calculate revenue and profit
        revenue = sales * self.price
        cost = sales * self.config.unit_cost
        profit = revenue - cost
        
        # Calculate inventory holding cost - always use dynamic calculation
        # Use predicted demand for next step as the forecast
        forecast_demand = self._get_demand_forecast()
        holding_cost_rate = self.config.calculate_holding_cost(
            inventory=self.inventory,
            demand_forecast=forecast_demand
        )
        
        inventory_cost = self.inventory * holding_cost_rate * self.config.unit_cost
        profit -= inventory_cost
        
        # Stockout penalty (if we couldn't meet demand)
        stockout = max(0, self.last_demand - sales)
        stockout_penalty = stockout * self.config.stockout_penalty
        profit -= stockout_penalty
        
        # Price stability penalty (if price changed too much)
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            price_change = abs(self.price - prev_price) / prev_price
            price_stability_penalty = self.gamma * price_change * self.config.unit_cost
            profit -= price_stability_penalty
        else:
            price_stability_penalty = 0
        
        # Excessive discount penalty (if price is much lower than unit cost)
        if self.price < self.config.unit_cost:
            discount_pct = (self.config.unit_cost - self.price) / self.config.unit_cost
            discount_penalty = self.delta * discount_pct * self.config.unit_cost
            profit -= discount_penalty
        else:
            discount_penalty = 0
        
        # Save profit
        self.profit_history.append(profit)
        
        # Calculate reward components
        revenue_component = self.alpha * revenue / (self.config.base_demand * self.config.unit_cost)
        
        # Market share component (our sales vs. total market)
        market_share = sales / (demand + 1e-6)  # Avoid division by zero
        market_share_component = self.beta * market_share
        
        # Inventory management component
        inventory_ratio = 1.0 if self._restock_inventory()[1] else self.inventory / (self.config.initial_inventory + 1e-6)
        inventory_component = self.gamma * (1.0 - abs(0.5 - inventory_ratio))
        
        # Profit margin component
        profit_margin = profit / (revenue + 1e-6)  # Avoid division by zero
        profit_margin_component = self.delta * max(0, profit_margin)
        
        # Combine reward components
        reward = revenue_component + market_share_component + inventory_component + profit_margin_component
        
        # Update time step
        self.time_step += 1
        
        # Check if episode is done
        self.done = self.time_step >= self.time_horizon
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary for monitoring
        info = {
            'price': self.price,
            'comp_price': self.comp_price,
            'demand': demand,
            'sales': sales,
            'revenue': revenue,
            'profit': profit,
            'inventory': self.inventory,
            'market_share': market_share,
            'stockout_penalty': stockout_penalty,
            'price_stability_penalty': price_stability_penalty,
            'discount_penalty': discount_penalty,
            'is_promotion': promo_flag,
            'reward_components': {
                'revenue': revenue_component,
                'market_share': market_share_component,
                'inventory': inventory_component,
                'profit_margin': profit_margin_component
            }
        }
        
        return next_state, reward, self.done, False, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode (human, rgb_array)
            
        Returns:
            Rendered frame if mode is rgb_array, None otherwise
        """
        # Simple text rendering
        if mode == 'human':
            print(f"Day: {self.time_step}")
            print(f"Price: ${self.price:.2f}")
            print(f"Competitor price: ${self.comp_price:.2f}")
            print(f"Demand: {self.last_demand:.1f}")
            print(f"Sales: {self.last_sales:.1f}")
            print(f"Inventory: {self.inventory:.1f}")
            if len(self.profit_history) > 0:
                print(f"Profit: ${self.profit_history[-1]:.2f}")
            print("----------------------------")
            return None
        else:
            return None
    
    def _get_demand_forecast(self) -> float:
        """
        Get a simple demand forecast for the next step.
        
        Returns:
            Forecasted demand
        """
        # Use average of last 3 demands if available, otherwise use last demand
        if len(self.demand_history) >= 3:
            return np.mean(self.demand_history[-3:])
        elif len(self.demand_history) > 0:
            return self.demand_history[-1]
        else:
            # Default forecast if no history is available
            return self.config.base_demand 