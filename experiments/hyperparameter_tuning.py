"""
Hyperparameter tuning for MSME Pricing RL project.

This module implements hyperparameter tuning for the reward weights and other 
key parameters to optimize agent behavior with a focus on our updated reward function.
"""

import os
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

# Import environment and agent
from environments.config import MSMEConfig
from environments.msme_env import MSMEEnvironment
from agents.dqn_agent import MSMEPricingAgent

def tune_hyperparameters(
    agent_type: str = 'dqn',
    config_name: str = 'default',
    num_trials: int = 30,
    num_episodes: int = 50,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run hyperparameter tuning for the updated reward function weights and environment parameters.
    
    Args:
        agent_type: Type of agent to tune ('dqn')
        config_name: Environment configuration name
        num_trials: Number of tuning trials to run
        num_episodes: Number of episodes to evaluate each trial
        output_dir: Directory to save results
        
    Returns:
        Dictionary of best hyperparameters
    """
    print(f"Starting hyperparameter tuning for {agent_type} agent with {config_name} config")
    print(f"Running {num_trials} trials with {num_episodes} episodes each")
    
    # Create Optuna study for maximizing profit and stability
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_type}_{config_name}_tuning"
    )
    
    # Define objective function for optimization
    def objective(trial):
        # Sample hyperparameters to tune
        
        # Reward weights for the updated reward function
        alpha = trial.suggest_float("alpha", 0.1, 0.5)     # Holding cost penalty weight
        beta = trial.suggest_float("beta", 0.2, 0.6)       # Stockout penalty weight
        gamma = trial.suggest_float("gamma", 0.1, 0.3)     # Price instability penalty weight
        delta = trial.suggest_float("delta", 0.3, 0.7)     # Discount penalty weight
        row = trial.suggest_float("row", 0.01, 0.1)        # Fill rate bonus weight
        
        # Create environment with sampled parameters
        config = MSMEConfig(config_path=f"config/{config_name}_config.yaml" if config_name != "default" else None)
        
        # Create environment
        env = MSMEEnvironment(
            config=config,
            alpha=alpha,   # Holding cost penalty weight
            beta=beta,     # Stockout penalty weight
            gamma=gamma,   # Price instability penalty weight
            delta=delta,   # Discount penalty weight
            row=row,       # Fill rate bonus weight
            use_data_driven_competitor=True
        )
        
        # Create agent
        agent = MSMEPricingAgent(
            env=env,
            gamma=trial.suggest_float("discount_factor", 0.9, 0.99),
            epsilon_decay=trial.suggest_float("epsilon_decay", 150, 250),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        )
        
        # Train agent
        rewards = []
        profits = []
        fill_rates = []
        price_stabilities = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_profit = 0
            price_changes = []
            episode_fill_rates = []
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Get action from agent
                action = agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # Track metrics
                episode_reward += reward
                if 'profit' in info:
                    episode_profit += info['profit']
                
                # Track price stability
                if len(env.price_history) >= 2:
                    prev_price = env.price_history[-2]
                    curr_price = env.price_history[-1]
                    price_change = abs(curr_price - prev_price)
                    price_changes.append(price_change)
                
                # Track fill rate
                if 'fill_rate' in info:
                    episode_fill_rates.append(info['fill_rate'])
                
                # Save transition to memory
                agent.memory.push(
                    torch.FloatTensor(state),
                    action,
                    torch.FloatTensor(next_state) if not done else None,
                    reward,
                    done
                )
                
                # Optimize model
                loss = agent.optimize_model()
                
                # Update state
                state = next_state
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # Track episode metrics
            agent.episodes_done += 1
            rewards.append(episode_reward)
            profits.append(episode_profit)
            
            # Calculate price stability (negative of average price change)
            if len(price_changes) > 0:
                price_stability = -np.mean(price_changes)
                price_stabilities.append(price_stability)
            
            # Calculate average fill rate
            if len(episode_fill_rates) > 0:
                avg_fill_rate = np.mean(episode_fill_rates)
                fill_rates.append(avg_fill_rate)
        
        # Calculate metrics for optimization
        avg_reward = np.mean(rewards[-10:])  # Average reward of last 10 episodes
        avg_profit = np.mean(profits[-10:])  # Average profit of last 10 episodes
        
        # Price stability score (higher is better)
        price_stability_score = np.mean(price_stabilities) if price_stabilities else -1.0
        
        # Fill rate score (higher is better)
        fill_rate_score = np.mean(fill_rates) if fill_rates else 0.0
        
        # Combined objective: balance profit, reward, price stability, and fill rate
        combined_score = 0.5 * avg_profit + 0.2 * avg_reward + 0.2 * price_stability_score + 0.1 * fill_rate_score
        
        return combined_score
    
    # Run optimization
    study.optimize(objective, n_trials=num_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Print best parameters
    print("\nBest Hyperparameters:")
    print(f"Reward Weights:")
    print(f"  alpha (holding cost penalty): {best_params['alpha']:.4f}")
    print(f"  beta (stockout penalty): {best_params['beta']:.4f}")
    print(f"  gamma (price instability penalty): {best_params['gamma']:.4f}")
    print(f"  delta (discount penalty): {best_params['delta']:.4f}")
    print(f"  row (fill rate bonus): {best_params['row']:.4f}")
    
    print(f"Agent Parameters:")
    print(f"  learning_rate: {best_params['learning_rate']:.6f}")
    print(f"  epsilon_decay: {best_params['epsilon_decay']:.4f}")
    print(f"  discount_factor: {best_params['discount_factor']:.4f}")
    
    print(f"\nBest Combined Score: {best_value:.4f}")
    
    # Save optimized config
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{agent_type}_{config_name}_optimized_params.txt")
        
        with open(output_path, 'w') as f:
            f.write("Best Hyperparameters:\n")
            f.write(f"Reward Weights:\n")
            f.write(f"  alpha (holding cost penalty): {best_params['alpha']:.4f}\n")
            f.write(f"  beta (stockout penalty): {best_params['beta']:.4f}\n")
            f.write(f"  gamma (price instability penalty): {best_params['gamma']:.4f}\n")
            f.write(f"  delta (discount penalty): {best_params['delta']:.4f}\n")
            f.write(f"  row (fill rate bonus): {best_params['row']:.4f}\n")
            
            f.write(f"Agent Parameters:\n")
            f.write(f"  learning_rate: {best_params['learning_rate']:.6f}\n")
            f.write(f"  epsilon_decay: {best_params['epsilon_decay']:.4f}\n")
            f.write(f"  discount_factor: {best_params['discount_factor']:.4f}\n")
            
            f.write(f"\nBest Combined Score: {best_value:.4f}\n")
        
        print(f"Saved optimized parameters to {output_path}")
        
        # Also create a YAML config file for easy loading in main training scripts
        yaml_path = os.path.join(output_dir, f"{agent_type}_{config_name}_reward_weights.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"# Optimized reward weights for {agent_type} agent with {config_name} config\n")
            f.write(f"alpha: {best_params['alpha']:.4f}  # Holding cost penalty weight\n")
            f.write(f"beta: {best_params['beta']:.4f}  # Stockout penalty weight\n")
            f.write(f"gamma: {best_params['gamma']:.4f}  # Price instability penalty weight\n")
            f.write(f"delta: {best_params['delta']:.4f}  # Discount penalty weight\n")
            f.write(f"row: {best_params['row']:.4f}  # Fill rate bonus weight\n")
        
        print(f"Saved optimized reward weights to {yaml_path}")
    
    return best_params

if __name__ == "__main__":
    # Run the tuning function
    best_params = tune_hyperparameters(
        agent_type='dqn',
        config_name='electronics',
        num_trials=20,
        num_episodes=30,
        output_dir='results/tuning'
    ) 