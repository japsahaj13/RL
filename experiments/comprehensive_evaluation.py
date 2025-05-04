#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for MSME Pricing RL Project

This script performs detailed evaluation of the trained RL agent against
traditional pricing methods, following the evaluation approach in research literature.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import datetime
import yaml
import scipy.optimize as optimize

from environments.config import MSMEConfig, load_config
from environments.msme_env import MSMEEnvironment
from agents.dqn_agent import MSMEPricingAgent

# Evaluation parameters
NUM_EVAL_EPISODES = 30
CONFIDENCE_INTERVAL = 0.95  # 95% confidence interval
VISUALIZE_RESULTS = True

class FixedMarginPricer:
    """Baseline pricer that sets prices based on a fixed margin above cost."""
    
    def __init__(self, cost: float, margin_pct: float = 0.3):
        """
        Initialize the fixed margin pricer.
        
        Args:
            cost: Unit cost of the product
            margin_pct: Margin percentage to add above cost (default: 30%)
        """
        self.cost = cost
        self.margin = margin_pct
        
    def get_price(self, state: np.ndarray) -> float:
        """
        Get price based on fixed margin.
        
        Args:
            state: Current state (not used)
            
        Returns:
            Price based on fixed margin
        """
        return self.cost * (1 + self.margin)

class CompetitorMatchPricer:
    """Baseline pricer that matches or slightly undercuts competitor prices."""
    
    def __init__(self, cost: float, undercut_pct: float = 0.05, min_margin_pct: float = 0.1):
        """
        Initialize the competitor price matcher.
        
        Args:
            cost: Unit cost of the product
            undercut_pct: Percentage to undercut competitor price (default: 5%)
            min_margin_pct: Minimum margin percentage to maintain (default: 10%)
        """
        self.cost = cost
        self.undercut_pct = undercut_pct
        self.min_margin = min_margin_pct
        
    def get_price(self, state: np.ndarray) -> float:
        """
        Get price based on competitor price with possible undercutting.
        
        Args:
            state: Current state [inventory, comp_price, last_demand, last_sales, price, time_step]
            
        Returns:
            Price based on competitor matching strategy
        """
        comp_price = state[1]  # Competitor price is at index 1
        
        # Calculate price with undercut
        target_price = comp_price * (1 - self.undercut_pct)
        
        # Ensure minimum margin
        min_price = self.cost * (1 + self.min_margin)
        return max(target_price, min_price)

class OptimizationBasedPricer:
    """Baseline pricer that uses traditional optimization to maximize profit."""
    
    def __init__(self, cost: float, price_elasticity: float, competitor_sensitivity: float, 
                 base_demand: float):
        """
        Initialize the optimization-based pricer.
        
        Args:
            cost: Unit cost of the product
            price_elasticity: Price elasticity of demand
            competitor_sensitivity: Sensitivity to competitor prices
            base_demand: Base demand at reference price
        """
        self.cost = cost
        self.price_elasticity = price_elasticity
        self.competitor_sensitivity = competitor_sensitivity
        self.base_demand = base_demand
    
    def estimate_demand(self, price: float, comp_price: float) -> float:
        """
        Estimate demand based on price and competitor price.
        
        Args:
            price: Our price
            comp_price: Competitor price
            
        Returns:
            Estimated demand
        """
        # Log-linear demand model
        price_effect = np.power(price, self.price_elasticity)
        competitor_effect = np.power(comp_price, self.competitor_sensitivity)
        demand = self.base_demand * price_effect * competitor_effect
        return demand
    
    def profit_function(self, price: float, comp_price: float) -> float:
        """
        Calculate profit for a given price and competitor price.
        
        Args:
            price: Our price
            comp_price: Competitor price
            
        Returns:
            Negative profit (for minimization)
        """
        demand = self.estimate_demand(price, comp_price)
        profit = (price - self.cost) * demand
        return -profit  # Return negative profit for minimization
    
    def get_price(self, state: np.ndarray) -> float:
        """
        Get optimal price based on profit maximization.
        
        Args:
            state: Current state [inventory, comp_price, last_demand, last_sales, price, time_step]
            
        Returns:
            Optimal price based on profit maximization
        """
        comp_price = state[1]  # Competitor price is at index 1
        
        # Define bounds and constraints
        min_price = self.cost * 1.05  # At least 5% margin
        max_price = self.cost * 2.0   # At most 100% margin
        bounds = [(min_price, max_price)]
        
        # Find optimal price
        result = optimize.minimize(
            lambda p: self.profit_function(p[0], comp_price),
            x0=[comp_price],  # Initial guess
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            return result.x[0]
        else:
            # Fallback to a reasonable price if optimization fails
            return comp_price * 0.95  # Slightly undercut competitor

def run_evaluation_episode(
    env: MSMEEnvironment, 
    agent: Any, 
    use_baseline: bool = False
) -> Tuple[Dict[str, float], List[float], List[float], List[float]]:
    """
    Run a single evaluation episode.
    
    Args:
        env: MSME environment
        agent: RL agent or baseline pricer
        use_baseline: Whether using a baseline pricer (not RL)
        
    Returns:
        Tuple of (metrics, prices, profits, demands)
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_profit = 0
    episode_sales = 0
    
    done = False
    truncated = False
    
    prices = []
    profits = []
    demands = []
    
    while not (done or truncated):
        # Select action based on agent type
        if use_baseline:
            # Convert action to price directly for baseline pricing methods
            price = agent.get_price(state)
            # Find closest price tier
            price_diffs = np.abs(env.config.price_tiers - price)
            action = np.argmin(price_diffs)
        else:
            # Use RL agent's greedy policy
            action = agent.select_action(state, evaluate=True)
        
        # Take step in environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Track metrics
        episode_reward += reward
        if 'profit' in info:
            episode_profit += info['profit']
        prices.append(env.price)
        profits.append(info.get('profit', 0))
        demands.append(env.last_demand)
        
        if env.last_sales > 0:
            episode_sales += env.last_sales
        
        # Update state
        state = next_state
    
    # Calculate metrics
    # Calculate fill rate
    total_demand = sum(demands)
    fill_rate = episode_sales / (total_demand + 1e-6) if total_demand > 0 else 0
    
    metrics = {
        'reward': episode_reward,
        'profit': episode_profit,
        'sales': episode_sales,
        'avg_price': np.mean(prices),
        'price_stability': -np.std(prices) / np.mean(prices),  # Negative coefficient of variation
        'profit_stability': -np.std(profits) / (np.mean(profits) + 1e-6),  # Avoid division by zero
        'fill_rate': fill_rate,  # Correctly calculated fill rate
        'inventory_efficiency': episode_sales / (env.config.initial_inventory + env.config.restock_amount)
    }
    
    return metrics, prices, profits, demands

def evaluate_agent(
    config_name: str,
    model_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of a trained RL agent.
    
    Args:
        config_name: Name of configuration to use
        model_path: Path to trained agent model
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating agent with config: {config_name}")
    
    # Load configuration
    config = load_config(config_name)
    
    # Create environment with standard weights
    # We should use the same weights as during training
    env = MSMEEnvironment(
        config=config,
        # Use standard reward weights
        alpha=0.2,     # Holding cost penalty weight
        beta=0.5,      # Stockout penalty weight
        gamma=0.6,     # Price stability penalty weight (increased from default 0.2)
        delta=0.2,     # Discount penalty weight
        row=0.1        # Fill rate bonus weight
    )
    
    # Create and load RL agent
    agent = MSMEPricingAgent(env=env)
    agent.load(model_path)
    
    # Create baseline pricing strategies
    fixed_margin = FixedMarginPricer(config.unit_cost, margin_pct=0.3)
    competitor_match = CompetitorMatchPricer(config.unit_cost, undercut_pct=0.05)
    optimization_based = OptimizationBasedPricer(
        cost=config.unit_cost,
        price_elasticity=config.price_elasticity,
        competitor_sensitivity=config.competitor_sensitivity,
        base_demand=config.base_demand
    )
    
    # Run evaluation episodes for RL agent
    rl_metrics_list = []
    rl_prices_list = []
    rl_profits_list = []
    rl_demands_list = []
    
    print(f"Running {NUM_EVAL_EPISODES} evaluation episodes for RL agent...")
    for i in range(NUM_EVAL_EPISODES):
        metrics, prices, profits, demands = run_evaluation_episode(env, agent)
        rl_metrics_list.append(metrics)
        rl_prices_list.append(prices)
        rl_profits_list.append(profits)
        rl_demands_list.append(demands)
        
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{NUM_EVAL_EPISODES} episodes")
    
    # Run evaluation episodes for baseline strategies
    baselines = {
        'Fixed_Margin': fixed_margin,
        'Competitor_Match': competitor_match,
        'Optimization': optimization_based
    }
    
    baseline_results = {}
    
    for name, baseline in baselines.items():
        print(f"Running {NUM_EVAL_EPISODES} evaluation episodes for {name} strategy...")
        metrics_list = []
        prices_list = []
        profits_list = []
        demands_list = []
        
        for i in range(NUM_EVAL_EPISODES):
            metrics, prices, profits, demands = run_evaluation_episode(env, baseline, use_baseline=True)
            metrics_list.append(metrics)
            prices_list.append(prices)
            profits_list.append(profits)
            demands_list.append(demands)
            
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{NUM_EVAL_EPISODES} episodes")
        
        baseline_results[name] = {
            'metrics': metrics_list,
            'prices': prices_list,
            'profits': profits_list,
            'demands': demands_list
        }
    
    # Calculate aggregate metrics with confidence intervals
    rl_aggregate = compute_aggregate_metrics(rl_metrics_list)
    baseline_aggregates = {}
    
    for name, results in baseline_results.items():
        baseline_aggregates[name] = compute_aggregate_metrics(results['metrics'])
    
    # Create comparison table
    comparison = {
        'RL_Agent': rl_aggregate
    }
    comparison.update(baseline_aggregates)
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table
        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'))
        
        # Create visualizations
        if VISUALIZE_RESULTS:
            create_visualizations(
                rl_prices_list, rl_profits_list, rl_demands_list,
                baseline_results,
                output_dir
            )
    
    return {
        'rl_agent': {
            'aggregate': rl_aggregate,
            'metrics': rl_metrics_list,
            'prices': rl_prices_list,
            'profits': rl_profits_list,
            'demands': rl_demands_list
        },
        'baselines': baseline_results,
        'comparison': comparison
    }

def compute_aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregate metrics with confidence intervals.
    
    Args:
        metrics_list: List of metrics dictionaries
        
    Returns:
        Dictionary with aggregate metrics
    """
    aggregate = {}
    
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        # Compute confidence interval
        ci = 1.96 * std / np.sqrt(n)  # 95% confidence interval
        
        aggregate[key] = {
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci
        }
    
    return aggregate

def create_visualizations(
    rl_prices_list: List[List[float]],
    rl_profits_list: List[List[float]],
    rl_demands_list: List[List[float]],
    baseline_results: Dict[str, Dict[str, Any]],
    output_dir: str
):
    """
    Create visualizations comparing RL agent and baseline strategies.
    
    Args:
        rl_prices_list: List of price histories for RL agent
        rl_profits_list: List of profit histories for RL agent
        rl_demands_list: List of demand histories for RL agent
        baseline_results: Dictionary with baseline results
        output_dir: Directory to save visualizations
    """
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 1. Create price stability comparison
    create_price_stability_visualization(rl_prices_list, baseline_results, output_dir)
    
    # 2. Create profit comparison
    create_profit_comparison_visualization(rl_profits_list, baseline_results, output_dir)
    
    # 3. Create demand comparison
    create_demand_comparison_visualization(rl_demands_list, baseline_results, output_dir)

def create_price_stability_visualization(
    rl_prices_list: List[List[float]],
    baseline_results: Dict[str, Dict[str, Any]],
    output_dir: str
):
    """Create price stability comparison visualization."""
    plt.figure(figsize=(12, 8))
    
    # Sample a single episode for visualization
    episode_idx = 0
    
    # Plot RL agent prices
    plt.plot(rl_prices_list[episode_idx], label='RL Agent', linewidth=2)
    
    # Plot baseline prices
    for name, results in baseline_results.items():
        plt.plot(results['prices'][episode_idx], label=name, linestyle='--')
    
    plt.title('Price Comparison: RL Agent vs. Baseline Strategies', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'price_comparison.png'), dpi=300)
    plt.close()

def create_profit_comparison_visualization(
    rl_profits_list: List[List[float]],
    baseline_results: Dict[str, Dict[str, Any]],
    output_dir: str
):
    """Create profit comparison visualization."""
    plt.figure(figsize=(12, 8))
    
    # Calculate cumulative profits
    rl_cum_profits = np.cumsum(rl_profits_list[0])
    
    # Plot RL agent cumulative profits
    plt.plot(rl_cum_profits, label='RL Agent', linewidth=2)
    
    # Plot baseline cumulative profits
    for name, results in baseline_results.items():
        cum_profits = np.cumsum(results['profits'][0])
        plt.plot(cum_profits, label=name, linestyle='--')
    
    plt.title('Cumulative Profit Comparison', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cumulative Profit', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'profit_comparison.png'), dpi=300)
    plt.close()

def create_demand_comparison_visualization(
    rl_demands_list: List[List[float]],
    baseline_results: Dict[str, Dict[str, Any]],
    output_dir: str
):
    """Create demand comparison visualization."""
    plt.figure(figsize=(12, 8))
    
    # Plot RL agent demands
    plt.plot(rl_demands_list[0], label='RL Agent', linewidth=2)
    
    # Plot baseline demands
    for name, results in baseline_results.items():
        plt.plot(results['demands'][0], label=name, linestyle='--')
    
    plt.title('Demand Comparison: RL Agent vs. Baseline Strategies', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Demand', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'demand_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of trained RL agents")
    parser.add_argument("--config", type=str, default="default",
                       help="Configuration name to use")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained agent model")
    parser.add_argument("--output", type=str, default=None,
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory if not provided
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("results", args.config, f"eval_{timestamp}")
    
    # Run evaluation
    results = evaluate_agent(
        config_name=args.config,
        model_path=args.model,
        output_dir=args.output
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 80)
    
    # Create a simple dataframe for console display
    metrics = ['reward', 'profit', 'sales', 'fill_rate', 'price_stability', 'profit_stability']
    summary = {}
    
    for method, data in results['comparison'].items():
        summary[method] = {metric: data[metric]['mean'] for metric in metrics}
    
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    print("\nDetailed results saved to:", args.output)
    print("=" * 80)

if __name__ == "__main__":
    main() 