# RL-Based Pricing Optimization for Retail

A reinforcement learning framework for optimizing pricing strategies in retail environments with enhanced reward function.

## Overview

This project implements data-driven pricing strategy optimization using reinforcement learning (RL). The system models a retail business environment where an agent needs to make dynamic pricing decisions based on:

- Inventory levels
- Competitor pricing 
- Demand elasticity
- Seasonal patterns
- Promotional events

The RL agent learns to maximize profit while balancing multiple business constraints by interacting with a simulated market environment.

## Reward Function

The environment uses a comprehensive reward function that balances multiple business objectives:

```
R = Profit − α × Holding Cost − β × Stockout Penalty − γ × Price Stability Penalty + row × Fill Rate Bonus
```

With an additional penalty if the price falls below 80% of the unit cost:
```
If price < 0.8 × unit_cost, then R = R − δ × Discount Penalty
```

Where:
- **Profit**: (price − unit_cost) × sales
- **Holding Cost**: inventory × holding_rate
- **Stockout Penalty**: unfulfilled_demand × stockout_rate
- **Price Stability Penalty**: |price − previous_price|
- **Fill Rate Bonus**: (sales / demand) × 100
- **Discount Penalty**: Activated for excessive discounting below cost

## Features

- **Enhanced Reward Function**: Balances profit, inventory costs, stockouts, price stability, and fill rate
- **Data-Driven Demand Modeling**: Log-log demand model fitted from real retail data
- **Multiple RL Algorithms**: DQN, A2C, and PPO implementations
- **Business-Relevant Environment**: Realistic market simulation with inventory management
- **Flexible Configuration**: Easy configuration for different product categories and scenarios
- **Interactive Visualization**: Tools to analyze and visualize pricing strategies
- **Hyperparameter Tuning**: Automated tuning for reward function weights

## Project Structure

```
RL/
├── data/                              # Data files
├── demand_modeling/                   # Demand model implementation
├── environments/                      # RL environment implementation
├── agents/                            # RL agent implementations 
├── models/                            # Neural network models
├── utilities/                         # Utility functions
├── experiments/                       # Training and evaluation scripts
├── notebooks/                         # Jupyter notebooks
├── config/                            # Configuration files
├── results/                           # Training results and tuned parameters
├── tests/                             # Test files
├── tune_rewards.py                    # Reward function tuning script
├── main.py                            # Main entry point
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-pricing.git
cd rl-pricing

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Train a DQN agent on the electronics configuration
python main.py train --agent dqn --config electronics

# Evaluate a trained model
python main.py evaluate --model results/electronics/dqn_<timestamp>/agent --episodes 10

# Run a demo visualization
python main.py demo --model results/electronics/dqn_<timestamp>/agent
```

### Tuning Reward Function Weights

```bash
# Tune the reward function weights for specific product category
python tune_rewards.py --config electronics --agent dqn --trials 30 --episodes 30
```

## The Demand Model

The project uses a log-log demand model that captures various price elasticities:

```
log(Units_Sold) = α + e⋅log(Price) + c⋅log(Competitor_Price) + β₁⋅Promotion + β₂⋅Seasonality + ...
```

Where:
- `α` is the base demand (varies by region/season)
- `e` is the own-price elasticity (typically negative)
- `c` is the competitor price sensitivity
- Various β coefficients capture other factors

## Agent Implementations

The project implements multiple RL algorithms:

- **DQN**: With double and dueling extensions for stable learning
- **A2C**: Actor-critic implementation for faster convergence
- **PPO**: Proximal Policy Optimization for sample efficiency

## Customizing for Your Business

To adapt the environment for your specific business:

1. Add your product data to `data/`
2. Create a custom configuration in `config/`
3. Run demand model fitting with your data
4. Tune the reward weights for your business needs using `tune_rewards.py`
5. Train an agent with your configuration

## License

MIT

## Acknowledgments

- The demand model is inspired by economic pricing models and elasticity research
- The RL implementations are based on best practices from the deep RL literature 