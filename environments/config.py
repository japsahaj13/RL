"""
Environment Configuration for MSME Pricing RL project.
This module contains the MSMEConfig class for configuring pricing environments.
"""

import numpy as np
import os
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union

# Try to import demand model fitting
try:
    from demand_modeling.demand_model_fitting import fit_demand_models
except ImportError:
    print("Warning: demand_model_fitting module not found. Using default demand model.")
    fit_demand_models = None

# Import holding rate utilities
try:
    from utilities.holding_rate import compute_holding_cost_rate, compute_dynamic_holding_cost
except ImportError:
    print("Warning: holding_rate module not found. Using static holding cost.")
    compute_holding_cost_rate = None
    compute_dynamic_holding_cost = None

class MSMEConfig:
    """
    Configuration class for MSME pricing environment.
    
    This class stores all parameters needed to configure a pricing environment
    including product details, pricing tiers, demand model parameters, inventory
    management settings, and more.
    """
    
    def __init__(
            self, 
            product_name: str = "Sample Product",
            product_category: str = "Electronics",  # Default to Electronics
            region: str = "North",
            unit_cost: float = 50,
            price_tiers: Optional[np.ndarray] = None,
            base_demand: float = 200,         # Default, will be overridden by fitted model
            # Inventory management parameters
            holding_cost: float = 0.05,
            initial_inventory: int = 150,
            restock_level: int = 30,
            restock_amount: int = 120,
            restock_period: int = 7,
            stockout_penalty: float = 5.0,
            # Parameters for the log-log demand model - will be set by fitted model
            price_elasticity: float = -0.8,
            competitor_sensitivity: float = 0.3,
            promotion_multiplier: float = 1.2,
            season_effects: Optional[Dict[str, float]] = None,
            weather_effects: Optional[Dict[str, float]] = None,
            region_effects: Optional[Dict[str, float]] = None,
            demand_noise_scale: float = 0.05,
            use_fitted_model: bool = True,    # Always use fitted model by default
            # Parameters for dynamic holding cost
            use_dynamic_holding_cost: bool = True,
            storage_fee: float = 0.15,       # FIXED: Reduced from 2.0 to realistic value
            annual_finance_rate: float = 0.12,
            periods_per_year: int = 12,
            spoilage_pct: float = 0.002,      # FIXED: Reduced from 0.02 to realistic value
            holding_rate_x0: float = 0.25,
            holding_rate_k: float = 10.0,
            use_linear_holding_cost: bool = False,  # NEW: Option to use linear model instead of logistic
            linear_base_rate: float = 0.05,   # NEW: Base rate for linear model
            linear_excess_penalty: float = 0.05,  # NEW: Additional rate for excess inventory
            config_path: Optional[str] = None
    ):
        """
        Initialize the MSMEConfig with parameters for the pricing environment.
        
        Args:
            product_name: Name of the product
            product_category: Category of the product (Electronics, Grocery, etc.)
            region: Region (North, South, East, West)
            unit_cost: Cost of producing one unit
            price_tiers: Available price points (if None, auto-generated)
            base_demand: Base demand at reference price (will be overridden by fitted model)
            holding_cost: Cost per unit of inventory per period
            initial_inventory: Starting inventory
            restock_level: Reorder when inventory drops below this
            restock_amount: Amount to restock
            restock_period: Periods between restock checks
            stockout_penalty: Penalty for stockouts
            price_elasticity: Price elasticity coefficient (will be set by fitted model)
            competitor_sensitivity: Competitor price sensitivity (will be set by fitted model)
            promotion_multiplier: Multiplier for promotion periods (will be set by fitted model)
            season_effects: Season-specific multipliers (will be set by fitted model)
            weather_effects: Weather-specific multipliers (will be set by fitted model)
            region_effects: Region-specific multipliers (will be set by fitted model)
            demand_noise_scale: Scale of random noise in demand
            use_fitted_model: Whether to use fitted model from data (default True)
            use_dynamic_holding_cost: Whether to use dynamic holding cost model
            storage_fee: Storage fee per unit per period
            annual_finance_rate: Annual finance rate
            periods_per_year: Number of periods in a year
            spoilage_pct: Spoilage percentage per period
            holding_rate_x0: Midpoint for holding rate logistic function
            holding_rate_k: Steepness of the holding rate curve
            use_linear_holding_cost: Whether to use linear holding cost model instead of logistic
            linear_base_rate: Base rate for linear holding cost model
            linear_excess_penalty: Additional rate for excess inventory in linear model
            config_path: Path to YAML configuration file to load
        """
        # If config path is provided, load from file
        if config_path is not None:
            self._load_from_yaml(config_path)
            return
            
        self.product_name = product_name
        self.product_category = product_category
        self.region = region
        self.unit_cost = unit_cost
        
        # Calculate price tiers as percentage changes from unit cost
        if price_tiers is None:
            # Define percentage changes
            # A = {-50%, -30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +30%}
            pct_changes = [
                -0.5, -0.3, -0.2, -0.1, 
                -0.05, 0.0, 0.05, 0.1, 
                0.2, 0.3
            ]
            percentage_changes = np.array(pct_changes)
            # Calculate price tiers as price = unit_cost * (1 + pct)
            self.price_tiers = np.array([
                unit_cost * (1 + pct) for pct in percentage_changes
            ])
        else:
            self.price_tiers = price_tiers
            
        # Initialize demand model parameters (will be overridden by fitted model)
        self.base_demand = base_demand
        self.price_elasticity = price_elasticity
        self.competitor_sensitivity = competitor_sensitivity
        self.promotion_multiplier = promotion_multiplier
        
        # Inventory management parameters
        self.holding_cost = holding_cost
        self.initial_inventory = initial_inventory
        self.restock_level = restock_level
        self.restock_amount = restock_amount
        self.restock_period = restock_period
        self.stockout_penalty = stockout_penalty
        
        # Default season effects if not provided (will be overridden by fitted model)
        if season_effects is None:
            self.season_effects = {
                'Spring': 1.0,
                'Summer': 1.2,
                'Autumn': 0.8,
                'Winter': 0.9
            }
        else:
            self.season_effects = season_effects
            
        # Default weather effects if not provided (will be overridden by fitted model)
        if weather_effects is None:
            self.weather_effects = {
                'Sunny': 1.0,
                'Cloudy': 0.9,
                'Rainy': 0.8,
                'Snowy': 0.7
            }
        else:
            self.weather_effects = weather_effects
            
        # Default region effects if not provided (will be overridden by fitted model)
        if region_effects is None:
            self.region_effects = {
                'North': 1.0,
                'South': 1.1,
                'East': 0.9,
                'West': 1.0
            }
        else:
            self.region_effects = region_effects
            
        self.demand_noise_scale = demand_noise_scale
        self.use_fitted_model = use_fitted_model
        
        # Dynamic holding cost parameters
        self.use_dynamic_holding_cost = use_dynamic_holding_cost
        self.storage_fee = storage_fee
        self.annual_finance_rate = annual_finance_rate
        self.periods_per_year = periods_per_year
        self.spoilage_pct = spoilage_pct
        self.holding_rate_x0 = holding_rate_x0
        self.holding_rate_k = holding_rate_k
        
        # Linear holding cost parameters
        self.use_linear_holding_cost = use_linear_holding_cost
        self.linear_base_rate = linear_base_rate
        self.linear_excess_penalty = linear_excess_penalty
        
        # Always load fitted model parameters (regardless of use_fitted_model)
        # This ensures all demand parameters are properly set from data
        try:
            self._load_fitted_model_params()
        except Exception as e:
            print(f"Error loading fitted model parameters: {e}")
            print("This is critical - we need to fit demand models from data!")
            # Force fitting new models from data if available
            if fit_demand_models is not None:
                print("Attempting to fit new demand models from dataset...")
                try:
                    model_params = fit_demand_models()
                    if self.product_category in model_params:
                        self._set_demand_params_from_model(model_params[self.product_category])
                    else:
                        print(f"Critical: No model available for {self.product_category}!")
                except Exception as fit_error:
                    print(f"Failed to fit new models: {fit_error}")
                    print("The environment will not work correctly without demand models!")
            else:
                print("Cannot fit demand models - demand_model_fitting not available!")
                print("The environment will not work correctly without demand models!")
                
        # Always calculate initial holding cost using the dynamic model
        # Even if use_dynamic_holding_cost is False - we'll always use dynamic calculation
        if compute_dynamic_holding_cost is not None:
            try:
                # Initialize with a reasonable default demand forecast
                self._calculate_initial_holding_cost()
            except Exception as e:
                print(f"Warning: Failed to calculate dynamic holding cost: {e}. Using default value.")
                # Use a reasonable default value based on the product category
                if self.product_category == "Groceries":
                    self.holding_cost = 0.08  # Higher for perishable goods
                elif self.product_category == "Electronics":
                    self.holding_cost = 0.06  # Higher for valuable items
                elif self.product_category == "Furniture":
                    self.holding_cost = 0.04  # Lower due to low spoilage but higher space needs
                else:
                    self.holding_cost = 0.05  # Default value
        else:
            print("Warning: dynamic holding cost calculation not available - using default value.")
            self.holding_cost = 0.05  # Default fallback
    
    def _set_demand_params_from_model(self, params):
        """Helper method to set all demand parameters from fitted model"""
        self.base_demand = params['base_demand']
        self.price_elasticity = params['price_elasticity']
        self.competitor_sensitivity = params['competitor_sensitivity']
        self.promotion_multiplier = params['promotion_multiplier']
        self.season_effects = params['season_effects']
        self.weather_effects = params['weather_effects']
        self.region_effects = params['region_effects']
        print(f"Demand parameters set for {self.product_category}:")
        print(f"- Base demand: {self.base_demand:.2f}")
        print(f"- Price elasticity: {self.price_elasticity:.2f}")
        print(f"- Competitor sensitivity: {self.competitor_sensitivity:.2f}")
        print(f"- Promotion multiplier: {self.promotion_multiplier:.2f}")
    
    def _load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set attributes from config file
        for key, value in config.items():
            setattr(self, key, value)
            
        # Convert price_tiers to numpy array if it exists
        if hasattr(self, 'price_tiers') and self.price_tiers is not None:
            self.price_tiers = np.array(self.price_tiers)
        
        # Always load fitted demand model params, even if loaded from YAML
        try:
            self._load_fitted_model_params()
        except Exception as e:
            print(f"Error loading fitted model parameters when loading from YAML: {e}")
            print("This is critical - the environment may not work correctly!")
        
    def _load_fitted_model_params(self) -> None:
        """
        Load fitted model parameters from demand_models.pkl
        
        This method loads parameters for the specified product category.
        """
        # First check in the project root
        pickle_paths = [
            'demand_models.pkl',
            '../demand_models.pkl',
            os.path.join(os.path.dirname(__file__), '..', 'demand_models.pkl'),
            'data/demand_models.pkl'
        ]
        
        model_params = None
        
        # Try each path
        for path in pickle_paths:
            if os.path.exists(path):
                print(f"Found demand models at {path}")
                with open(path, 'rb') as f:
                    model_params = pickle.load(f)
                break
        
        # If not found, try to fit models from dataset
        if model_params is None and fit_demand_models is not None:
            print("No demand models found. Fitting models from dataset...")
            model_params = fit_demand_models()
        
        # If still None, raise exception
        if model_params is None:
            raise FileNotFoundError("Could not find or create demand models")
            
        # If the product category exists in the fitted models
        if self.product_category in model_params:
            self._set_demand_params_from_model(model_params[self.product_category])
        else:
            available_categories = ", ".join(model_params.keys())
            raise ValueError(f"No fitted model found for category '{self.product_category}'. Available categories: {available_categories}")
    
    def save_to_yaml(self, file_path: str) -> None:
        """
        Save the configuration to a YAML file
        
        Args:
            file_path: Path to save the YAML file
        """
        # Convert numpy arrays to lists for YAML serialization
        config_dict = self.__dict__.copy()
        if 'price_tiers' in config_dict and isinstance(config_dict['price_tiers'], np.ndarray):
            config_dict['price_tiers'] = config_dict['price_tiers'].tolist()
            
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """
        Return a string representation of the configuration
        """
        return (f"Product: {self.product_name}\n"
                f"Category: {self.product_category}\n"
                f"Region: {self.region}\n"
                f"Unit Cost: ${self.unit_cost}\n"
                f"Base Demand: {self.base_demand:.2f}\n"
                f"Price Elasticity: {self.price_elasticity:.2f}\n"
                f"Competitor Sensitivity: {self.competitor_sensitivity:.2f}\n"
                f"Promotion Multiplier: {self.promotion_multiplier:.2f}")

    def _calculate_initial_holding_cost(self):
        """
        Calculate the initial holding cost using the dynamic model.
        """
        # Use 80% of initial inventory as a reasonable default demand forecast
        default_demand_forecast = self.initial_inventory * 0.8
        
        self.holding_cost = compute_dynamic_holding_cost(
            inventory=self.initial_inventory,
            demand_forecast=default_demand_forecast,
            unit_cost=self.unit_cost,
            storage_fee=self.storage_fee,
            annual_finance_rate=self.annual_finance_rate,
            periods_per_year=self.periods_per_year,
            spoilage_pct=self.spoilage_pct,
            x0=self.holding_rate_x0,
            k=self.holding_rate_k
        )
        
        print(f"Initial dynamic holding cost calculated: {self.holding_cost:.4f}")
    
    def calculate_holding_cost(self, inventory: float, demand_forecast: float) -> float:
        """
        Calculate holding cost based on current inventory and demand forecast.
        
        Args:
            inventory: Current inventory level
            demand_forecast: Forecasted demand
            
        Returns:
            Calculated holding cost rate
        """
        if inventory <= 0:
            return 0.0
        
        # Calculate Excess Inventory Rate (EIR)
        eir = max(0, (inventory - demand_forecast) / inventory)
        
        # If linear model is selected, use a simple linear function
        if hasattr(self, 'use_linear_holding_cost') and self.use_linear_holding_cost:
            base_rate = self.linear_base_rate if hasattr(self, 'linear_base_rate') else 0.05
            excess_penalty = self.linear_excess_penalty if hasattr(self, 'linear_excess_penalty') else 0.05
            return base_rate + eir * excess_penalty
        
        # Otherwise, use the original dynamic or static holding cost calculation
        if compute_dynamic_holding_cost is not None and self.use_dynamic_holding_cost:
            try:
                return compute_dynamic_holding_cost(
                    inventory=inventory,
                    demand_forecast=demand_forecast,
                    unit_cost=self.unit_cost,
                    storage_fee=self.storage_fee,
                    annual_finance_rate=self.annual_finance_rate,
                    periods_per_year=self.periods_per_year,
                    spoilage_pct=self.spoilage_pct,
                    x0=self.holding_rate_x0,
                    k=self.holding_rate_k
                )
            except Exception as e:
                print(f"Warning: Dynamic holding cost calculation failed: {e}")
                # Fall back to pre-calculated holding cost
                return self.holding_cost
        else:
            # Fall back to pre-calculated holding cost
            return self.holding_cost


def create_default_config() -> MSMEConfig:
    """
    Create a default configuration
    
    Returns:
        Default MSMEConfig object
    """
    return MSMEConfig(product_category="Electronics")


def create_groceries_config() -> MSMEConfig:
    """
    Create a configuration for the Groceries category
    
    Returns:
        MSMEConfig object for Groceries
    """
    return MSMEConfig(
        product_name="Grocery Item",
        product_category="Groceries",
        region="North",
        unit_cost=10,
        initial_inventory=400,
        restock_level=100,
        restock_amount=300,
        restock_period=3
    )


def create_toys_config() -> MSMEConfig:
    """
    Create a configuration for the Toys category
    
    Returns:
        MSMEConfig object for Toys
    """
    return MSMEConfig(
        product_name="Toy Item",
        product_category="Toys",
        region="East",
        unit_cost=25,
        initial_inventory=200,
        restock_level=50,
        restock_amount=150,
        restock_period=7
    )


def create_electronics_config() -> MSMEConfig:
    """
    Create a configuration for the Electronics category
    
    Returns:
        MSMEConfig object for Electronics
    """
    return MSMEConfig(
        product_name="Electronics Item",
        product_category="Electronics",
        region="North",
        unit_cost=75,
        initial_inventory=150,
        restock_level=40,
        restock_amount=100,
        restock_period=10
    )


def create_furniture_config() -> MSMEConfig:
    """
    Create a configuration for the Furniture category
    
    Returns:
        MSMEConfig object for Furniture
    """
    return MSMEConfig(
        product_name="Furniture Item",
        product_category="Furniture",
        region="West",
        unit_cost=150,
        initial_inventory=100,
        restock_level=20,
        restock_amount=50,
        restock_period=14
    )


def create_clothing_config() -> MSMEConfig:
    """
    Create a configuration for the Clothing category
    
    Returns:
        MSMEConfig object for Clothing
    """
    return MSMEConfig(
        product_name="Clothing Item",
        product_category="Clothing",
        region="South",
        unit_cost=40,
        initial_inventory=200,
        restock_level=50,
        restock_amount=150,
        restock_period=7
    )


if __name__ == "__main__":
    # Generate category-specific configurations
    os.makedirs("../config", exist_ok=True)
    
    create_default_config().save_to_yaml("../config/default_config.yaml")
    create_groceries_config().save_to_yaml("../config/groceries_config.yaml")
    create_toys_config().save_to_yaml("../config/toys_config.yaml") 
    create_electronics_config().save_to_yaml("../config/electronics_config.yaml")
    create_furniture_config().save_to_yaml("../config/furniture_config.yaml")
    create_clothing_config().save_to_yaml("../config/clothing_config.yaml")
    
    print("Category-specific configurations saved to config directory.") 