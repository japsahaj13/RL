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
    from utilities.data_driven_parameters import get_restock_parameters, get_unit_cost, get_all_parameters
    from utilities.restock_prediction import predict_restock_parameters, get_current_restock_parameters
except ImportError:
    print("Warning: utilities modules not found. Using static parameters.")
    compute_holding_cost_rate = None
    compute_dynamic_holding_cost = None
    get_restock_parameters = None
    get_unit_cost = None
    get_all_parameters = None
    predict_restock_parameters = None
    get_current_restock_parameters = None

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
            unit_cost: float = 40,         # CHANGED: Reduced from 50 to improve profit margin
            price_tiers: Optional[np.ndarray] = None,
            base_demand: float = 200,         # Default, will be overridden by fitted model
            # Inventory management parameters
            holding_cost: float = 0.05,
            initial_inventory: int = 80,    # Further reduced to lower initial investment
            restock_level: int = 30,        # CHANGED: Reduced to avoid excessive restocking
            restock_amount: int = 100,      # CHANGED: Reduced to minimize capital tied up in inventory
            restock_period: int = 6,        # CHANGED: Adjusted for balance
            stockout_penalty: float = 1.0,   # CHANGED: Further reduced to 1.0
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
            use_dynamic_holding_cost: bool = False,  # Changed to False to prefer linear model
            storage_fee: float = 0.10,       # FIXED: Reduced from 0.15 to even more realistic value
            annual_finance_rate: float = 0.12,
            periods_per_year: int = 12,
            spoilage_pct: float = 0.001,     # FIXED: Reduced from 0.002 to more realistic value
            holding_rate_x0: float = 0.25,
            holding_rate_k: float = 10.0,
            use_linear_holding_cost: bool = True,  # CHANGED: Always use linear model by default
            linear_base_rate: float = 0.02,   # OPTIMIZED: Lower base rate (was 0.05)
            linear_excess_penalty: float = 0.03,  # OPTIMIZED: Lower penalty (was 0.05)
            use_dynamic_restock: bool = False,   # Whether to use dynamic restock parameter prediction
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
            use_dynamic_restock: Whether to use dynamic restock parameter prediction
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
            # Define percentage changes with higher margins
            # A = {-30%, -20%, -10%, 0%, +10%, +20%, +30%, +40%, +50%, +60%}
            pct_changes = [
                -0.3, -0.2, -0.1, 0.0, 
                0.1, 0.2, 0.3, 0.4, 
                0.5, 0.6
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
        
        # Dynamic restock parameters
        self.use_dynamic_restock = use_dynamic_restock
        
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
        try:
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
        except Exception as e:
            print(f"Error loading YAML config file {config_path}: {e}")
            print("Falling back to default configuration")
            # Use default configuration for the given product category
            if hasattr(self, 'product_category'):
                category = self.product_category
            else:
                category = "Electronics"  # Default category
                
            if category == "Groceries":
                config_obj = create_groceries_config()
            elif category == "Toys":
                config_obj = create_toys_config()
            elif category == "Furniture":
                config_obj = create_furniture_config()
            elif category == "Clothing":
                config_obj = create_clothing_config()
            else:  # Default to Electronics
                config_obj = create_default_config()
                
            # Copy attributes from the created config object
            for key, value in config_obj.__dict__.items():
                setattr(self, key, value)
    
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
            base_rate = self.linear_base_rate if hasattr(self, 'linear_base_rate') else 0.02
            excess_penalty = self.linear_excess_penalty if hasattr(self, 'linear_excess_penalty') else 0.03
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

    @property
    def should_use_dynamic_restock(self) -> bool:
        """
        Check if dynamic restock prediction should be used.
        
        Returns:
            Whether to use dynamic restock parameter prediction
        """
        return hasattr(self, 'use_dynamic_restock') and self.use_dynamic_restock and predict_restock_parameters is not None
        
    def get_restock_parameters(self, current_time: int, demand_forecast: float = None, 
                              price: float = None, units_sold: float = None) -> Dict[str, int]:
        """
        Get restock parameters, either static or dynamically predicted.
        
        Args:
            current_time: Current time step in simulation
            demand_forecast: Current demand forecast (optional)
            price: Current price (optional)
            units_sold: Recent units sold (optional)
            
        Returns:
            Dictionary with restock parameters
        """
        # If dynamic restock is not enabled or the module is not available, return static parameters
        if not self.should_use_dynamic_restock:
            return {
                'restock_period': self.restock_period,
                'restock_level': self.restock_level,
                'restock_amount': self.restock_amount
            }
            
        # Use current parameter values if not provided
        if demand_forecast is None:
            demand_forecast = self.base_demand
        if price is None:
            price = self.price_tiers[len(self.price_tiers) // 2]  # Middle price tier
        if units_sold is None:
            units_sold = self.base_demand * 0.8  # Default units sold
            
        # Try to predict new parameters if period has been exhausted
        new_params = predict_restock_parameters(
            category=self.product_category,
            demand_forecast=demand_forecast,
            inventory_level=getattr(self, '_current_inventory', self.initial_inventory),
            price=price,
            units_sold=units_sold,
            current_time=current_time
        )
        
        # If new parameters were predicted, update the config
        if new_params is not None:
            self.restock_period = new_params['restock_period']
            self.restock_level = new_params['restock_level']
            self.restock_amount = new_params['restock_amount']
            return new_params
            
        # Otherwise, get current parameters
        return get_current_restock_parameters(self.product_category)


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
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for Groceries
    """
    # Get data-driven parameters
    if get_all_parameters is not None:
        params = get_all_parameters("Groceries")
        initial_inventory = params["initial_inventory"]
        restock_level = params["restock_level"]
        restock_amount = params["restock_amount"] 
        restock_period = params["restock_period"]
        unit_cost = params["unit_cost"]
    else:
        # Use default values from previous analysis if module not available
        initial_inventory = 262
        restock_level = 166
        restock_amount = 110
        restock_period = 5
        unit_cost = 10
    
    return MSMEConfig(
        product_name="Groceries Item",
        product_category="Groceries",
        region="North",
        unit_cost=unit_cost,
        initial_inventory=initial_inventory,
        restock_level=restock_level,
        restock_amount=restock_amount,
        restock_period=restock_period,
        use_dynamic_restock=True  # Enable dynamic restock prediction
    )


def create_toys_config() -> MSMEConfig:
    """
    Create a configuration for the Toys category
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for Toys
    """
    # Get data-driven parameters
    if get_all_parameters is not None:
        params = get_all_parameters("Toys")
        initial_inventory = params["initial_inventory"]
        restock_level = params["restock_level"]
        restock_amount = params["restock_amount"] 
        restock_period = params["restock_period"]
        unit_cost = params["unit_cost"]
    else:
        # Use default values from previous analysis if module not available
        initial_inventory = 240
        restock_level = 164
        restock_amount = 110
        restock_period = 5
        unit_cost = 25
    
    return MSMEConfig(
        product_name="Toys Item",
        product_category="Toys",
        region="East",
        unit_cost=unit_cost,
        initial_inventory=initial_inventory,
        restock_level=restock_level,
        restock_amount=restock_amount,
        restock_period=restock_period,
        use_dynamic_restock=True  # Enable dynamic restock prediction
    )


def create_electronics_config() -> MSMEConfig:
    """
    Create a configuration for the Electronics category
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for Electronics
    """
    # Get data-driven parameters
    if get_all_parameters is not None:
        params = get_all_parameters("Electronics")
        initial_inventory = params["initial_inventory"]
        restock_level = params["restock_level"]
        restock_amount = params["restock_amount"] 
        restock_period = params["restock_period"]
        unit_cost = params["unit_cost"]
    else:
        # Use default values from previous analysis if module not available
        initial_inventory = 278
        restock_level = 163
        restock_amount = 110
        restock_period = 5
        unit_cost = 75
    
    return MSMEConfig(
        product_name="Electronics Item",
        product_category="Electronics",
        region="North",
        unit_cost=unit_cost,
        initial_inventory=initial_inventory,
        restock_level=restock_level,
        restock_amount=restock_amount,
        restock_period=restock_period,
        use_dynamic_restock=True  # Enable dynamic restock prediction
    )


def create_furniture_config() -> MSMEConfig:
    """
    Create a configuration for the Furniture category
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for Furniture
    """
    # Get data-driven parameters
    if get_all_parameters is not None:
        params = get_all_parameters("Furniture")
        initial_inventory = params["initial_inventory"]
        restock_level = params["restock_level"]
        restock_amount = params["restock_amount"] 
        restock_period = params["restock_period"]
        unit_cost = params["unit_cost"]
    else:
        # Use default values from previous analysis if module not available
        initial_inventory = 278
        restock_level = 166
        restock_amount = 110
        restock_period = 5
        unit_cost = 150
    
    return MSMEConfig(
        product_name="Furniture Item",
        product_category="Furniture",
        region="West",
        unit_cost=unit_cost,
        initial_inventory=initial_inventory,
        restock_level=restock_level,
        restock_amount=restock_amount,
        restock_period=restock_period,
        use_dynamic_restock=True  # Enable dynamic restock prediction
    )


def create_clothing_config() -> MSMEConfig:
    """
    Create a configuration for the Clothing category
    with data-driven inventory parameters.
    
    Returns:
        MSMEConfig object for Clothing
    """
    # Get data-driven parameters
    if get_all_parameters is not None:
        params = get_all_parameters("Clothing")
        initial_inventory = params["initial_inventory"]
        restock_level = params["restock_level"]
        restock_amount = params["restock_amount"] 
        restock_period = params["restock_period"]
        unit_cost = params["unit_cost"]
    else:
        # Use default values from previous analysis if module not available
        initial_inventory = 229
        restock_level = 164
        restock_amount = 110
        restock_period = 5
        unit_cost = 40
    
    return MSMEConfig(
        product_name="Clothing Item",
        product_category="Clothing",
        region="South",
        unit_cost=unit_cost,
        initial_inventory=initial_inventory,
        restock_level=restock_level,
        restock_amount=restock_amount,
        restock_period=restock_period,
        use_dynamic_restock=True  # Enable dynamic restock prediction
    )


def load_config(config_name: str) -> MSMEConfig:
    """
    Load a configuration by name.
    
    Args:
        config_name: Name of the configuration to load (without _config.yaml suffix)
        
    Returns:
        MSMEConfig object
    """
    config_path = os.path.join("config", f"{config_name}_config.yaml")
    if not os.path.exists(config_path):
        # If config doesn't exist with name, try without suffix
        config_path = os.path.join("config", f"{config_name}.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
        
    return MSMEConfig(config_path=config_path)


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