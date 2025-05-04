#!/usr/bin/env python3
"""
Reward Function Hyperparameter Tuning

This script runs hyperparameter tuning for the reward weights of our modified reward function
for different product categories.
"""

import os
import argparse
import sys
from pathlib import Path

# Ensure modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from experiments.hyperparameter_tuning import tune_hyperparameters


def main():
    """Run the hyperparameter tuning for reward weights."""
    parser = argparse.ArgumentParser(description="Tune reward weights for MSME Pricing environment")
    parser.add_argument("--config", type=str, default="electronics",
                        choices=["default", "electronics", "groceries", "toys", "furniture", "clothing"],
                        help="Product category to tune")
    parser.add_argument("--agent", type=str, default="dqn",
                        choices=["dqn", "a2c", "ppo"],
                        help="Agent type to tune")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of hyperparameter tuning trials")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Number of episodes to run per trial")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join("results", "tuning")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting hyperparameter tuning for {args.agent} agent with {args.config} config")
    print(f"Running {args.trials} trials with {args.episodes} episodes per trial")
    
    # Run tuning
    best_params = tune_hyperparameters(
        agent_type=args.agent,
        config_name=args.config,
        num_trials=args.trials,
        num_episodes=args.episodes,
        output_dir=output_dir
    )
    
    print("\nTuning completed!")
    print("Best hyperparameters:")
    print(f"  alpha (holding cost penalty): {best_params['alpha']:.4f}")
    print(f"  beta (stockout penalty): {best_params['beta']:.4f}")
    print(f"  gamma (price instability penalty): {best_params['gamma']:.4f}")
    print(f"  delta (discount penalty): {best_params['delta']:.4f}")
    print(f"  row (fill rate bonus): {best_params['row']:.4f}")


if __name__ == "__main__":
    main() 