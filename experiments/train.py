"""
Training experiments for RL agents on the MSME Pricing environment.
"""

import os
import argparse
import torch
import numpy as np
import datetime
import yaml
from typing import Dict, List, Optional, Any, Tuple

from environments.config import (
    MSMEConfig, 
    create_electronics_config, 
    create_groceries_config,
    create_toys_config,
    create_furniture_config,
    create_clothing_config
)
from environments.msme_env import MSMEEnvironment
from agents.dqn_agent import MSMEPricingAgent

def load_config(config_name: str) -> MSMEConfig:
    """
    Load a configuration based on its name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        MSMEConfig object
    """
    # Check for predefined configurations
    if config_name == 'default':
        return MSMEConfig()
    elif config_name == 'electronics':
        return create_electronics_config()
    elif config_name == 'groceries':
        return create_groceries_config()
    elif config_name == 'toys':
        return create_toys_config()
    elif config_name == 'furniture':
        return create_furniture_config()
    elif config_name == 'clothing':
        return create_clothing_config()
    
    # Try to load from YAML file
    config_path = f"config/{config_name}_config.yaml"
    if os.path.exists(config_path):
        config = MSMEConfig(config_path=config_path)
        return config
    
    # Fallback to default config
    print(f"Configuration '{config_name}' not found. Using default configuration.")
    return MSMEConfig()

def load_reward_weights(config_name: str, agent_type: str = 'dqn') -> Dict[str, float]:
    """
    Load optimized reward weights from YAML file if available.
    
    For daily pricing, the weights are adjusted to be proportional to daily rewards
    rather than monthly rewards.
    
    Args:
        config_name: Name of the configuration
        agent_type: Type of agent (dqn, a2c, ppo)
        
    Returns:
        Dictionary of reward weights or default values
    """
    # Check for optimized weights file
    weights_path = os.path.join("results/tuning", f"{agent_type}_{config_name}_reward_weights.yaml")
    
    # Updated default weights for daily rewards (scaled appropriately from monthly values)
    default_weights = {
        "alpha": 0.2,    # Holding cost penalty weight (monthly rate converted to daily in environment)
        "beta": 0.3,     # Stockout penalty weight (proportional to daily lost sales)
        "gamma": 0.4,    # Price instability penalty (important for daily price decisions)
        "delta": 0.3,    # Discount penalty weight (scaled for daily pricing)
        "row": 0.1       # Fill rate bonus weight (scaled for daily rewards)
    }
    
    # If optimized weights file exists, load it
    if os.path.exists(weights_path):
        try:
            with open(weights_path, 'r') as f:
                weights = yaml.safe_load(f)
                print(f"Loaded optimized reward weights from {weights_path}")
                return weights
        except Exception as e:
            print(f"Error loading reward weights: {e}")
            print("Using default weights instead")
    else:
        print(f"No optimized weights found at {weights_path}")
        print("Using default weights for daily pricing")
    
    return default_weights

def train_dqn(
        config_name: str = 'default',
        num_episodes: int = 300,
        save_path: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None
) -> Dict[str, List]:
    """
    Train a DQN agent on the MSME Pricing environment.
    
    The environment uses a daily pricing approach where:
    - Each episode represents a month (30 days)
    - Agent makes pricing decisions daily
    - Daily holding costs and penalties are applied
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes (months)
        save_path: Path to save the trained agent
        reward_weights: Optional dictionary of reward weights to use
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Load reward weights (optimized if available)
    if reward_weights is None:
        weights = load_reward_weights(config_name, "dqn")
    else:
        weights = reward_weights
        print("Using provided reward weights")
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        time_horizon=30,  # Set time horizon to 30 days to represent a monthly pricing period
        # Use loaded reward weights - adjusted for daily pricing
        alpha=weights["alpha"],     # Holding cost penalty weight
        beta=weights["beta"],       # Stockout penalty weight
        gamma=weights["gamma"],     # Price instability penalty weight
        delta=weights["delta"],     # Discount penalty weight
        row=weights["row"]          # Fill rate bonus weight
    )
    
    # Create agent
    agent = MSMEPricingAgent(
        env=env,
        gamma=0.99,  # Higher discount factor for daily rewards
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=200,
        learning_rate=0.001,
        batch_size=64,
        hidden_size=128,
        target_update=10,
        memory_size=10000,
        use_double_dqn=True,
        use_dueling=True
    )
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", config_name, f"dqn_{timestamp}_daily")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
    # Save reward weights used
    with open(os.path.join(output_dir, "reward_weights.yaml"), 'w') as f:
        yaml.dump(weights, f)
    
    # Train the agent
    training_metrics = agent.train(
        num_episodes=num_episodes,
        print_every=10,
        visualization_dir=os.path.join(output_dir, "visualizations")
    )
    
    # Save the agent
    if save_path is None:
        save_path = os.path.join(output_dir, "agent")
    agent.save(save_path)
    
    # Evaluate the agent
    eval_metrics = agent.evaluate(num_episodes=10)
    
    # Save metrics
    metrics = {
        "training": training_metrics,
        "evaluation": eval_metrics
    }
    
    # Save metrics as YAML
    with open(os.path.join(output_dir, "metrics.yaml"), 'w') as f:
        yaml.dump(metrics, f)
    
    return training_metrics

def train_a2c(
        config_name: str = 'default',
        num_episodes: int = 300,
        save_path: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None
) -> Dict[str, List]:
    """
    Train an A2C agent on the MSME Pricing environment.
    
    The environment uses a daily pricing approach where:
    - Each episode represents a month (30 days)
    - Agent makes pricing decisions daily
    - Daily holding costs and penalties are applied
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes (months)
        save_path: Path to save the trained agent
        reward_weights: Optional dictionary of reward weights to use
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Load reward weights (optimized if available)
    if reward_weights is None:
        weights = load_reward_weights(config_name, "a2c")
    else:
        weights = reward_weights
        print("Using provided reward weights")
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        time_horizon=30,  # Set time horizon to 30 days to represent a monthly pricing period
        # Use loaded reward weights - adjusted for daily pricing
        alpha=weights["alpha"],     # Holding cost penalty weight
        beta=weights["beta"],       # Stockout penalty weight
        gamma=weights["gamma"],     # Price instability penalty weight
        delta=weights["delta"],     # Discount penalty weight
        row=weights["row"]          # Fill rate bonus weight
    )
    
    # Import A2C agent
    from agents.a2c_agent import A2CPricingAgent
    
    # Create agent
    agent = A2CPricingAgent(
        env=env,
        gamma=0.99,  # Higher discount factor for daily rewards
        learning_rate=0.001,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        hidden_size=128
    )
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", config_name, f"a2c_{timestamp}_daily")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
    # Save reward weights used
    with open(os.path.join(output_dir, "reward_weights.yaml"), 'w') as f:
        yaml.dump(weights, f)
    
    # Train the agent
    training_metrics = agent.train(
        num_episodes=num_episodes,
        print_every=10,
        visualization_dir=os.path.join(output_dir, "visualizations")
    )
    
    # Save the agent
    if save_path is None:
        save_path = os.path.join(output_dir, "agent")
    agent.save(save_path)
    
    # Evaluate the agent
    eval_metrics = agent.evaluate(num_episodes=10)
    
    # Save metrics
    metrics = {
        "training": training_metrics,
        "evaluation": eval_metrics
    }
    
    # Save metrics as YAML
    with open(os.path.join(output_dir, "metrics.yaml"), 'w') as f:
        yaml.dump(metrics, f)
    
    return training_metrics

def train_ppo(
        config_name: str = 'default',
        num_episodes: int = 300,
        save_path: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None
) -> Dict[str, List]:
    """
    Train a PPO agent on the MSME Pricing environment.
    
    The environment uses a daily pricing approach where:
    - Each episode represents a month (30 days)
    - Agent makes pricing decisions daily
    - Daily holding costs and penalties are applied
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes (months)
        save_path: Path to save the trained agent
        reward_weights: Optional dictionary of reward weights to use
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Load reward weights (optimized if available)
    if reward_weights is None:
        weights = load_reward_weights(config_name, "ppo")
    else:
        weights = reward_weights
        print("Using provided reward weights")
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        time_horizon=30,  # Set time horizon to 30 days to represent a monthly pricing period
        # Use loaded reward weights - adjusted for daily pricing
        alpha=weights["alpha"],     # Holding cost penalty weight
        beta=weights["beta"],       # Stockout penalty weight
        gamma=weights["gamma"],     # Price instability penalty weight
        delta=weights["delta"],     # Discount penalty weight
        row=weights["row"]          # Fill rate bonus weight
    )
    
    # Import PPO agent
    from agents.ppo_agent import PPOPricingAgent
    
    # Create agent
    agent = PPOPricingAgent(
        env=env,
        gamma=0.99,  # Higher discount factor for daily rewards
        learning_rate=0.001,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        hidden_size=128
    )
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", config_name, f"ppo_{timestamp}_daily")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
    # Save reward weights used
    with open(os.path.join(output_dir, "reward_weights.yaml"), 'w') as f:
        yaml.dump(weights, f)
    
    # Train the agent
    training_metrics = agent.train(
        num_episodes=num_episodes,
        print_every=10,
        visualization_dir=os.path.join(output_dir, "visualizations")
    )
    
    # Save the agent
    if save_path is None:
        save_path = os.path.join(output_dir, "agent")
    agent.save(save_path)
    
    # Evaluate the agent
    eval_metrics = agent.evaluate(num_episodes=10)
    
    # Save metrics
    metrics = {
        "training": training_metrics,
        "evaluation": eval_metrics
    }
    
    # Save metrics as YAML
    with open(os.path.join(output_dir, "metrics.yaml"), 'w') as f:
        yaml.dump(metrics, f)
    
    return training_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on the MSME Pricing environment")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "a2c", "ppo"],
                        help="RL algorithm to use")
    parser.add_argument("--config", type=str, default="default",
                        help="Configuration name to use")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Number of training episodes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the trained agent")
    
    args = parser.parse_args()
    
    if args.agent == "dqn":
        train_dqn(args.config, args.episodes, args.output)
    elif args.agent == "a2c":
        train_a2c(args.config, args.episodes, args.output)
    elif args.agent == "ppo":
        train_ppo(args.config, args.episodes, args.output) 