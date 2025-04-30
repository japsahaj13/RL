"""
Evaluation scripts for trained RL agents on the MSME Pricing environment.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
from typing import Dict, List, Optional, Any, Tuple
import datetime
import time

from environments.config import MSMEConfig
from environments.msme_env import MSMEEnvironment
from agents.dqn_agent import MSMEPricingAgent
from experiments.train import load_config

def evaluate_model(
        model_path: str,
        num_episodes: int = 10,
        render: bool = False
) -> Dict[str, float]:
    """
    Evaluate a trained agent on the MSME Pricing environment.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract configuration path from the model path
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.yaml")
    
    # Load configuration if available, else use default
    if os.path.exists(config_path):
        config = MSMEConfig(config_path=config_path)
    else:
        print(f"Configuration file not found at {config_path}. Using default configuration.")
        config = MSMEConfig()
    
    # Create environment
    env = MSMEEnvironment(config)
    
    # Create agent
    agent = MSMEPricingAgent(env=env)
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    # Evaluate the agent
    eval_metrics = agent.evaluate(num_episodes=num_episodes, render=render)
    
    # Save metrics
    output_dir = os.path.join(model_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(os.path.join(output_dir, f"eval_metrics_{timestamp}.yaml"), 'w') as f:
        yaml.dump(eval_metrics, f)
    
    return eval_metrics

def run_demo(
        model_path: str,
        save_video: bool = False
) -> None:
    """
    Run a demonstration of a trained agent.
    
    Args:
        model_path: Path to the trained model
        save_video: Whether to save a video of the demonstration
    """
    # Extract configuration path from the model path
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.yaml")
    
    # Load configuration if available, else use default
    if os.path.exists(config_path):
        config = MSMEConfig(config_path=config_path)
    else:
        print(f"Configuration file not found at {config_path}. Using default configuration.")
        config = MSMEConfig()
    
    # Create environment
    env = MSMEEnvironment(config)
    
    # Create agent
    agent = MSMEPricingAgent(env=env)
    
    # Load the trained model
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory for visualization
    output_dir = os.path.join(model_dir, "demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run a single episode and collect data
    state, _ = env.reset()
    done = False
    
    episode_prices = []
    episode_comp_prices = []
    episode_demands = []
    episode_sales = []
    episode_inventories = []
    episode_profits = []
    
    while not done:
        # Select action (no exploration)
        action = agent.select_action(state, evaluate=True)
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render to console
        env.render()
        time.sleep(0.2)  # Slow down the demo for visualization
        
        # Store information for visualization
        episode_prices.append(info['price'])
        episode_comp_prices.append(info['comp_price'])
        episode_demands.append(info['demand'])
        episode_sales.append(info['sales'])
        episode_inventories.append(info['inventory'])
        episode_profits.append(info['profit'])
        
        # Move to next state
        state = next_state
    
    # Visualize the episode
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.visualize_results({
        'episode': timestamp,
        'prices': episode_prices,
        'comp_prices': episode_comp_prices,
        'demands': episode_demands,
        'sales': episode_sales,
        'inventories': episode_inventories,
        'profits': episode_profits
    }, save_path=os.path.join(output_dir, f"demo_{timestamp}.png"))
    
    # Calculate statistics
    total_profit = sum(episode_profits)
    avg_price = np.mean(episode_prices)
    avg_comp_price = np.mean(episode_comp_prices)
    avg_demand = np.mean(episode_demands)
    avg_sales = np.mean(episode_sales)
    final_inventory = episode_inventories[-1]
    
    print("\nDemo Results:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Price: ${avg_price:.2f}")
    print(f"Average Competitor Price: ${avg_comp_price:.2f}")
    print(f"Average Demand: {avg_demand:.2f}")
    print(f"Average Sales: {avg_sales:.2f}")
    print(f"Final Inventory: {final_inventory:.2f}")
    
    # Save the statistics
    stats = {
        'total_profit': float(total_profit),
        'avg_price': float(avg_price),
        'avg_comp_price': float(avg_comp_price),
        'avg_demand': float(avg_demand),
        'avg_sales': float(avg_sales),
        'final_inventory': float(final_inventory)
    }
    
    with open(os.path.join(output_dir, f"demo_stats_{timestamp}.yaml"), 'w') as f:
        yaml.dump(stats, f)
    
    print(f"Demo results saved to {output_dir}")

def compare_agents(
        model_paths: List[str],
        num_episodes: int = 5
) -> None:
    """
    Compare multiple agents on the same environment.
    
    Args:
        model_paths: List of paths to trained models
        num_episodes: Number of evaluation episodes per agent
    """
    # Use the configuration from the first model
    model_dir = os.path.dirname(model_paths[0])
    config_path = os.path.join(model_dir, "config.yaml")
    
    if os.path.exists(config_path):
        config = MSMEConfig(config_path=config_path)
    else:
        print(f"Configuration file not found at {config_path}. Using default configuration.")
        config = MSMEConfig()
    
    # Create environment
    env = MSMEEnvironment(config)
    
    # Create output directory
    output_dir = os.path.join("results", "comparisons")
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    for model_path in model_paths:
        # Create agent
        agent = MSMEPricingAgent(env=env)
        
        # Extract model name
        model_name = os.path.basename(model_path)
        
        # Load the trained model
        try:
            agent.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
        
        # Evaluate the agent
        print(f"Evaluating {model_name}...")
        eval_metrics = agent.evaluate(num_episodes=num_episodes)
        
        # Store metrics
        all_metrics[model_name] = eval_metrics
    
    # Save comparison results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f"comparison_{timestamp}.yaml"), 'w') as f:
        yaml.dump(all_metrics, f)
    
    # Create comparison visualizations
    create_comparison_plots(all_metrics, os.path.join(output_dir, f"comparison_{timestamp}.png"))
    
    print(f"Comparison results saved to {output_dir}")

def create_comparison_plots(metrics: Dict[str, Dict[str, float]], save_path: str) -> None:
    """
    Create plots comparing multiple agents.
    
    Args:
        metrics: Dictionary of metrics for each agent
        save_path: Path to save the plot
    """
    # Extract model names
    model_names = list(metrics.keys())
    
    # Extract metrics for comparison
    rewards = [metrics[name]['avg_reward'] for name in model_names]
    profits = [metrics[name]['avg_profit'] for name in model_names]
    sales = [metrics[name]['avg_sales'] for name in model_names]
    lengths = [metrics[name]['avg_length'] for name in model_names]
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Rewards
    plt.subplot(2, 2, 1)
    plt.bar(model_names, rewards)
    plt.title('Average Reward')
    plt.ylabel('Reward')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Profits
    plt.subplot(2, 2, 2)
    plt.bar(model_names, profits)
    plt.title('Average Profit')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Sales
    plt.subplot(2, 2, 3)
    plt.bar(model_names, sales)
    plt.title('Average Sales')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Episode lengths
    plt.subplot(2, 2, 4)
    plt.bar(model_names, lengths)
    plt.title('Average Episode Length')
    plt.ylabel('Length')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.suptitle('Agent Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on the MSME Pricing environment")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--demo", action="store_true",
                        help="Run a demonstration of the agent")
    parser.add_argument("--save-video", action="store_true",
                        help="Save a video of the demonstration")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo(args.model, args.save_video)
    else:
        evaluate_model(args.model, args.episodes, args.render) 