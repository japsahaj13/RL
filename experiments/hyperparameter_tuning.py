"""
Hyperparameter tuning for MSME Pricing RL project.

This module implements hyperparameter tuning for the reward weights and other 
key parameters to optimize agent behavior, particularly to reduce price volatility
and improve profit stability.
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
    Run hyperparameter tuning for reward weights and environment parameters.
    
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
    
    # Create Optuna study for maximizing profit stability and reward
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{agent_type}_{config_name}_tuning"
    )
    
    # Define objective function for optimization
    def objective(trial):
        # Sample hyperparameters to tune
        
        # Reward weights
        alpha = trial.suggest_float("alpha", 0.1, 0.5)  # Revenue weight
        beta = trial.suggest_float("beta", 0.1, 0.7)    # Market share weight
        gamma = trial.suggest_float("gamma", 0.2, 0.8)  # Inventory/price stability weight
        delta = trial.suggest_float("delta", 0.1, 0.7)  # Profit margin weight
        
        # Normalize weights to sum to 1.0
        total = alpha + beta + gamma + delta
        alpha /= total
        beta /= total
        gamma /= total
        delta /= total
        
        # Create environment with sampled parameters
        config = MSMEConfig(config_path=f"config/{config_name}_config.yaml" if config_name != "default" else None)
        
        # Create environment
        env = MSMEEnvironment(
            config=config,
            alpha=alpha,   # Revenue weight
            beta=beta,     # Market share weight
            gamma=gamma,   # Inventory/price stability weight
            delta=delta,   # Profit margin weight
            use_data_driven_competitor=True
        )
        
        # Create agent
        agent = MSMEPricingAgent(
            env=env,
            gamma=trial.suggest_float("discount_factor", 0.9, 0.99),
            epsilon_decay=trial.suggest_float("epsilon_decay", 0.9, 0.995) * num_episodes,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        )
        
        # Train agent
        rewards = []
        profit_stabilities = []
        price_stabilities = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            price_changes = []
            profits = []
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Get action from agent
                action = agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, truncated, info = env.step(action)
                
                # Save price change and profit
                if len(env.price_history) >= 2:
                    prev_price = env.price_history[-2]
                    curr_price = env.price_history[-1]
                    price_change = abs(curr_price - prev_price) / prev_price
                    price_changes.append(price_change)
                
                if 'profit' in info:
                    profits.append(info['profit'])
                
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
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            # Update target network
            if episode % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # Track episode metrics
            agent.episodes_done += 1
            rewards.append(episode_reward)
            
            # Calculate price stability (negative of average price change)
            if len(price_changes) > 0:
                price_stability = -np.mean(price_changes)
                price_stabilities.append(price_stability)
            
            # Calculate profit stability (negative of profit standard deviation)
            if len(profits) > 1:
                profit_stability = -np.std(profits)
                profit_stabilities.append(profit_stability)
        
        # Calculate metrics for optimization
        avg_reward = np.mean(rewards[-10:])  # Average reward of last 10 episodes
        
        # Price stability score (higher is better)
        price_stability_score = np.mean(price_stabilities) if price_stabilities else -1.0
        
        # Profit stability score (higher is better)
        profit_stability_score = np.mean(profit_stabilities) if profit_stabilities else -1.0
        
        # Combined objective: balance reward and stability
        combined_score = 0.6 * avg_reward + 0.2 * price_stability_score + 0.2 * profit_stability_score
        
        return combined_score
    
    # Run optimization
    study.optimize(objective, n_trials=num_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Normalize weights to sum to 1.0
    weights_sum = (
        best_params["alpha"] + 
        best_params["beta"] + 
        best_params["gamma"] + 
        best_params["delta"]
    )
    
    best_params["alpha"] /= weights_sum
    best_params["beta"] /= weights_sum
    best_params["gamma"] /= weights_sum
    best_params["delta"] /= weights_sum
    
    # Print best parameters
    print("\nBest Hyperparameters:")
    print(f"Reward Weights:")
    print(f"  alpha (revenue): {best_params['alpha']:.4f}")
    print(f"  beta (market share): {best_params['beta']:.4f}")
    print(f"  gamma (inventory/price stability): {best_params['gamma']:.4f}")
    print(f"  delta (profit margin): {best_params['delta']:.4f}")
    
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
            f.write(f"  alpha (revenue): {best_params['alpha']:.4f}\n")
            f.write(f"  beta (market share): {best_params['beta']:.4f}\n")
            f.write(f"  gamma (inventory/price stability): {best_params['gamma']:.4f}\n")
            f.write(f"  delta (profit margin): {best_params['delta']:.4f}\n")
            
            f.write(f"Agent Parameters:\n")
            f.write(f"  learning_rate: {best_params['learning_rate']:.6f}\n")
            f.write(f"  epsilon_decay: {best_params['epsilon_decay']:.4f}\n")
            f.write(f"  discount_factor: {best_params['discount_factor']:.4f}\n")
            
            f.write(f"\nBest Combined Score: {best_value:.4f}\n")
        
        print(f"Saved optimized parameters to {output_path}")
    
    return best_params

if __name__ == "__main__":
    # Test the tuning function
    best_params = tune_hyperparameters(
        agent_type='dqn',
        config_name='default',
        num_trials=15,
        num_episodes=15,
        output_dir='results/tuning'
    ) 