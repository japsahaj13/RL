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
            row: float = 0.1, use_data_driven_competitor: bool = True
    ):
        """
        Initialize the MSME pricing environment.
        
        Args:
            config: Configuration object with environment parameters
            time_horizon: Number of time steps for each episode
            alpha: Weight for holding cost penalty in reward function
            beta: Weight for stockout penalty in reward function
            gamma: Weight for price instability penalty in reward function
            delta: Weight for discount penalty in reward function
            row: Weight for fill rate bonus in reward function
            use_data_driven_competitor: Whether to use data-driven competitor pricing model
        """
        self.config = config
        self.time_horizon = time_horizon
        
        # Reward shaping weights
        self.alpha = alpha  # Holding cost penalty weight
        self.beta = beta    # Stockout penalty weight
        self.gamma = gamma  # Price instability penalty weight
        self.delta = delta  # Discount penalty weight
        self.row = row      # Fill rate bonus weight
        
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
        
        # Initialize with middle price tier for sampling competitor price
        # This will be used only to get an initial competitor price
        initial_price = self.config.price_tiers[len(self.config.price_tiers) // 2]
        
        # Sample competitor price first
        self.comp_price = self._sample_competitor_price(initial_price)
        
        # Then set agent's initial price (will be overridden by first action)
        self.price = initial_price
        
        # Calculate an initial demand estimate for inventory decisions
        promo_flag = self._is_promotion_period()
        self.last_demand = self._demand_model(self.price, self.comp_price, promo_flag, 0)
        
        self.last_sales = 0
        self.time_step = 0
        self.done = False
        
        # Reset history
        self.sales_history = []
        self.price_history = []
        self.demand_history = []
        self.inventory_history = []
        self.comp_price_history = [self.comp_price]  # Initialize with starting comp price
        self.profit_history = []
        self.price_change_count = 0  # Reset price change counter
        self._last_restock_cost = 0.0  # Initialize restock tracking
        self._last_did_restock = False
        
        # Set the price decision flag to True for first step
        # Now we'll make price decisions daily
        self._need_price_decision = True
        
        # Reset restock scheduling
        if hasattr(self.config, 'get_restock_parameters'):
            # Get initial parameters
            initial_params = self.config.get_restock_parameters(
                current_time=0,
                demand_forecast=self.last_demand,  # Use initial demand calculation
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
        # Get competitor's previous price if available, otherwise use our_price
        prev_comp_price = self.comp_price if hasattr(self, 'comp_price') and self.comp_price > 0 else our_price
        
        # Try to use the data-driven competitor model if enabled
        if self.use_data_driven_competitor:
            try:
                # Get current state variables
                discount = 0  # Default discount (could calculate from price_tiers)
                is_promotion = self._is_promotion_period()
                
                # Get competitor price prediction from the model
                # Use competitor's previous price instead of agent's price
                competitor_price = predict_competitor_price(
                    our_price=prev_comp_price,  # Use competitor's previous price instead
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
        else:
            if not hasattr(self, '_competitor_warning_shown'):
                print(f"Warning: Using simulated (not data-driven) competitor pricing for {self.config.product_category}")
                print("Consider generating and using competitor_models.pkl for better behavior.")
                self._competitor_warning_shown = True
        
        # Simulation-based competitor behavior (modified implementation)
        # Define valid price range for competitor
        min_price = self.config.unit_cost * 0.8
        max_price = self.config.unit_cost * 2.0
        
        # Competitor reacts based on its own previous price rather than ours
        if len(self.comp_price_history) > 0:
            # Competitor has a 10% chance of changing price each day
            if random.random() < 0.1:
                # Get the competitor's own previous price
                comp_previous_price = self.comp_price_history[-1]
                
                # Random strategy with minimal awareness of our price
                strategy = random.random()
                
                # Occasionally glance at our price, but usually follow own pricing strategy
                if strategy < 0.15 and len(self.price_history) > 0:  # Only 15% chance to look at our price
                    our_previous_price = self.price_history[-1]
                    if our_previous_price < comp_previous_price * 0.9:  # Our price is much lower
                        new_price = comp_previous_price * (0.92 + 0.03 * random.random())
                    elif our_previous_price > comp_previous_price * 1.1:  # Our price is much higher
                        new_price = comp_previous_price * (1.03 + 0.02 * random.random())
                    else:  # Our price is similar
                        new_price = comp_previous_price * (0.99 + 0.02 * random.random())
                else:
                    # Mostly follow their own pricing trends
                    if strategy < 0.6:
                        # Price decrease (market pressure, normal fluctuation)
                        new_price = comp_previous_price * (0.96 + 0.03 * random.random())
                    else:
                        # Price increase (inflation, cost increases)
                        new_price = comp_previous_price * (1.01 + 0.02 * random.random())
                
                # Ensure price is within valid range
                self.comp_price = max(min_price, min(max_price, new_price))
        
        # Add some random walk to the competitor price (reduced from 0.02 to 0.01)
        random_factor = 1.0 + 0.01 * (random.random() - 0.5)
        self.comp_price *= random_factor
        
        # Ensure price is within valid range
        return max(min_price, min(max_price, self.comp_price))
    
    def _restock_inventory(self) -> Tuple[float, bool]:
        """
        Handle inventory restocking logic.
        
        The method follows a periodic review policy:
        1. Check inventory level every restock_period days
        2. If inventory is below restock_level, order restock_amount units
        3. Restock_level is dynamically calculated based on expected demand
        
        Returns:
            Tuple of (restocking cost, whether restock occurred)
        """
        # Store current inventory for logging
        self.config._current_inventory = self.inventory
        
        # Initialize next_check_time if not set
        if not hasattr(self, 'next_restock_check_time'):
            # Get parameters from config/model
            params = self.config.get_restock_parameters(
                current_time=self.time_step,
                demand_forecast=self.last_demand if hasattr(self, 'last_demand') else None,
                price=self.price if hasattr(self, 'price') else None,
                units_sold=self.last_sales if hasattr(self, 'last_sales') else None
            )
            # Schedule first check based on returned period
            self.next_restock_check_time = params['restock_period']
        
        # Check if it's time to review inventory
        if self.time_step >= self.next_restock_check_time:
            # Get fresh parameters from model with current data
            params = self.config.get_restock_parameters(
                current_time=self.time_step,
                demand_forecast=self.last_demand,
                price=self.price,
                units_sold=self.last_sales
            )
            
            # Extract parameters directly from model output
            restock_level = self.config.restock_level
            restock_amount = params['restock_amount']
            restock_period = params['restock_period']
            
            # Use latest demand for dynamic restock level calculation
            latest_demand = self.last_demand if self.last_demand > 0 else self.config.base_demand
            
            # Calculate dynamic restock level based on demand * period + safety buffer
            safety_buffer = 0.2  # 20% safety stock
            dynamic_restock_level = latest_demand * restock_period * (1 + safety_buffer)
            
            # Use the higher of configured restock_level and dynamic_restock_level
            effective_restock_level = max(restock_level, dynamic_restock_level)
            
            # Check if inventory is below the restock level
            if self.inventory < effective_restock_level:
                # Add model-provided restock amount to inventory
                self.inventory += restock_amount
                
                # Store restock amount for info
                self._restock_amount = restock_amount
                
                # Calculate cost (fixed order cost plus handling)
                fixed_order_cost = 10.0  # Fixed cost per order
                handling_cost_per_unit = 0.02 * self.config.unit_cost  # 2% of unit cost for handling
                restock_cost = fixed_order_cost + (restock_amount * handling_cost_per_unit)
                
                # Schedule next check using model-provided period
                self.next_restock_check_time = self.time_step + restock_period
                
                return restock_cost, True
            else:
                # Schedule next check using model-provided period
                self.next_restock_check_time = self.time_step + restock_period
        
        # No restocking
        return 0.0, False
    
    def _calculate_base_price_demand(self) -> float:
        """
        Calculate demand at base price for fill rate calculations.
        
        The base price is typically the middle tier price.
        This method allows comparing actual demand (at current price)
        with what demand would be at a reference price point.
        
        Returns:
            Expected demand at base price
        """
        # Use middle tier price as reference price
        base_price = self.config.price_tiers[len(self.config.price_tiers) // 2]
        
        # Calculate demand at base price using same conditions
        promo_flag = self._is_promotion_period()
        base_demand = self._demand_model(base_price, self.comp_price, promo_flag, self.time_step)
        
        return base_demand
    
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
        
        # Set the price if a new price decision is needed
        if self._need_price_decision:
            old_price = self.price if len(self.price_history) > 0 else None
            self.price = self.config.price_tiers[action]
            self.price_history.append(self.price)
            self._need_price_decision = False  # Reset flag after using it
        
        # Calculate demand based on price and other factors (using the CURRENT price)
        # This demand calculation depends on the price set by the agent
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
        
        # Calculate inventory holding cost
        forecast_demand = self._get_demand_forecast()
        
        # Convert monthly holding cost rate to daily rate
        monthly_holding_cost_rate = self.config.calculate_holding_cost(
            inventory=self.inventory,
            demand_forecast=forecast_demand
        )
        
        # Convert monthly rate to daily rate (divide by 30 days in a month)
        daily_holding_cost_rate = monthly_holding_cost_rate / 30.0
        
        # Apply daily holding cost
        inventory_cost = self.inventory * daily_holding_cost_rate * self.config.unit_cost
        
        # Stockout penalty (if we couldn't meet demand)
        stockout = max(0, self.last_demand - sales)
        
        # Daily stockout penalty (adjusted to be proportional)
        stockout_penalty = stockout * self.config.stockout_penalty
        
        # Price stability penalty (focusing on frequency of changes)
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            # Apply penalty if price changed at all (binary penalty for any change)
            if self.price != prev_price:
                # Count how many price changes have occurred in this episode
                change_count = 1
                if hasattr(self, 'price_change_count'):
                    self.price_change_count += 1
                    change_count = self.price_change_count
                else:
                    self.price_change_count = 1
                
                # Daily price stability penalty - reduced from monthly values
                # Base penalty of 2.0 plus 1.0 for each additional change
                price_stability_penalty = 2.0 + (1.0 * (change_count - 1))
            else:
                price_stability_penalty = 0
        else:
            price_stability_penalty = 0
            self.price_change_count = 0
            
        # Calculate fill rate bonus based on demand at base price
        # This allows comparing actual sales with what would be expected at base price
        fill_rate_bonus = 0
        base_price_demand = self._calculate_base_price_demand()
        
        if base_price_demand > 0:
            # Calculate fill rate as sales divided by base price demand
            # This measures how well we're meeting the market demand at reference price
            fill_rate = min(sales / base_price_demand, 1.0)
            fill_rate_bonus = fill_rate * 10  # Scaled for daily basis
        
        # Apply promotion effect if active
        if promo_flag:
            # Demand is already adjusted, this is just for information
            self.promo_multiplier = self.config.promotion_multiplier
        
        # Initial reward calculation
        reward = profit - self.alpha * inventory_cost - self.beta * stockout_penalty - self.gamma * price_stability_penalty + self.row * fill_rate_bonus
        
        # Excessive discount penalty (if price falls below 80% of unit cost)
        discount_penalty = 0
        if self.price < 0.8 * self.config.unit_cost:
            # Penalty increases as price falls further below 80% of unit cost
            # Scaled down for daily basis
            discount_penalty = (0.8 * self.config.unit_cost - self.price) * 10
            reward = reward - self.delta * discount_penalty
        
        # Calculate raw profit without penalties for tracking
        raw_profit = profit - inventory_cost - stockout_penalty
        
        # Apply restock costs if applicable - but don't deduct from profit since this is investment in inventory
        restock_cost = 0
        if hasattr(self, '_restock_this_step') and self._restock_this_step:
            restock_cost = self._restock_amount * self.config.restock_cost
            # We'll track restock costs separately but not penalize profit directly
            # raw_profit -= restock_cost  # Commented out - don't reduce profit for inventory investment
            # Reset flag
            self._restock_this_step = False
        
        # Save profit (the raw profit for historical tracking)
        self.profit_history.append(raw_profit)
        
        # Perform inventory check and possible restocking
        restock_cost, did_restock = self._restock_inventory()
        
        # If restocking happened, record it but don't directly adjust profit
        if did_restock:
            # We'll track restock costs separately but not penalize profit/reward directly
            # This is because restocking is an investment in future sales, not an expense
            # raw_profit -= restock_cost  # Commented out
            # reward -= restock_cost      # Commented out
            
            # Record the restock info for the info dictionary later
            self._last_restock_cost = restock_cost
            self._last_did_restock = True
        else:
            self._last_restock_cost = 0.0
            self._last_did_restock = False
        
        # Update time step
        self.time_step += 1
        
        # Check if episode is done
        self.done = self.time_step >= self.time_horizon
        
        # Update competitor price AFTER the step is done, so it's ready for the NEXT step
        # Sample competitor price for the next step if episode is not done
        if not self.done:
            self.comp_price = self._sample_competitor_price(self.price)
            self.comp_price_history.append(self.comp_price)
            
            # Set the flag to make a new price decision in the next step
            self._need_price_decision = True
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary for monitoring
        info = {
            'price': self.price,
            'comp_price': self.comp_price,
            'demand': demand,
            'base_price_demand': base_price_demand,
            'sales': sales,
            'revenue': revenue,
            'pure_profit': profit,  # Pure economic profit (revenue - COGS)
            'profit': raw_profit,   # This is raw profit including operational costs
            'inventory': self.inventory,
            'fill_rate': fill_rate_bonus / 10 if fill_rate_bonus > 0 else 0,
            'stockout_penalty': stockout_penalty,
            'price_stability_penalty': price_stability_penalty,
            'price_changes': getattr(self, 'price_change_count', 0),
            'discount_penalty': discount_penalty,
            'holding_cost': inventory_cost,
            'is_promotion': promo_flag,
            'restock_cost': getattr(self, '_last_restock_cost', 0.0),
            'did_restock': getattr(self, '_last_did_restock', False),
            'reward_components': {
                'base_profit': profit,
                'holding_cost_penalty': -self.alpha * inventory_cost,
                'stockout_penalty': -self.beta * stockout_penalty,
                'price_stability_penalty': -self.gamma * price_stability_penalty,
                'fill_rate_bonus': self.row * fill_rate_bonus,
                'discount_penalty': -self.delta * discount_penalty if self.price < 0.8 * self.config.unit_cost else 0
            },
            # Add detailed components for easier debugging
            'unit_cost': self.config.unit_cost,
            'holding_cost_rate': daily_holding_cost_rate,
            'monthly_holding_cost_rate': monthly_holding_cost_rate,
            'restock_amount': getattr(self, '_restock_amount', 0),
            'restock_level': self.config.restock_level,
            'dynamic_restock_level': self.last_demand * self.config.restock_period * 1.2 if hasattr(self, 'last_demand') else 0,
            'cogs': cost,
            'stockout_qty': stockout
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