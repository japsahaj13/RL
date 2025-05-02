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

def train_dqn(
        config_name: str = 'default',
        num_episodes: int = 300,
        save_path: Optional[str] = None
) -> Dict[str, List]:
    """
    Train a DQN agent on the MSME Pricing environment.
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes
        save_path: Path to save the trained agent
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        # Use optimized reward weights from hyperparameter tuning
        alpha=0.1319,  # Revenue weight (reduced from 0.2)
        beta=0.2316,   # Market share weight (reduced from 0.5)
        gamma=0.3654,  # Inventory/price stability weight (increased from 0.2)
        delta=0.2712   # Profit margin weight (increased from 0.2)
    )
    
    # Create agent
    agent = MSMEPricingAgent(
        env=env,
        gamma=0.95,
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
    output_dir = os.path.join("results", config_name, f"dqn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
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
        save_path: Optional[str] = None
) -> Dict[str, List]:
    """
    Train an A2C agent on the MSME Pricing environment.
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes
        save_path: Path to save the trained agent
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        # Use optimized reward weights from hyperparameter tuning
        alpha=0.1319,  # Revenue weight (reduced from 0.2)
        beta=0.2316,   # Market share weight (reduced from 0.5)
        gamma=0.3654,  # Inventory/price stability weight (increased from 0.2)
        delta=0.2712   # Profit margin weight (increased from 0.2)
    )
    
    # Create agent
    agent = MSMEPricingAgent(
        env=env,
        gamma=0.95,
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
    output_dir = os.path.join("results", config_name, f"a2c_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
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
        save_path: Optional[str] = None
) -> Dict[str, List]:
    """
    Train a PPO agent on the MSME Pricing environment.
    
    Args:
        config_name: Name of the configuration to use
        num_episodes: Number of training episodes
        save_path: Path to save the trained agent
        
    Returns:
        Dictionary of training metrics
    """
    # Load configuration
    config = load_config(config_name)
    
    # Create environment
    env = MSMEEnvironment(
        config=config,
        # Use optimized reward weights from hyperparameter tuning
        alpha=0.1319,  # Revenue weight (reduced from 0.2)
        beta=0.2316,   # Market share weight (reduced from 0.5)
        gamma=0.3654,  # Inventory/price stability weight (increased from 0.2)
        delta=0.2712   # Profit margin weight (increased from 0.2)
    )
    
    # Create agent
    agent = MSMEPricingAgent(
        env=env,
        gamma=0.95,
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
    output_dir = os.path.join("results", config_name, f"ppo_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config.save_to_yaml(os.path.join(output_dir, "config.yaml"))
    
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