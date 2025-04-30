# RL-Based Pricing Optimization for Retail

A reinforcement learning framework for optimizing pricing strategies in retail environments.

## Overview

This project implements data-driven pricing strategy optimization using reinforcement learning (RL). The system models a retail business environment where an agent needs to make dynamic pricing decisions based on:

- Inventory levels
- Competitor pricing 
- Demand elasticity
- Seasonal patterns
- Promotional events

The RL agent learns to maximize profit while balancing multiple business constraints by interacting with a simulated market environment.

## Features

- **Data-Driven Demand Modeling**: Log-log demand model fitted from real retail data
- **Multiple RL Algorithms**: DQN, A2C, and PPO implementations
- **Business-Relevant Environment**: Realistic market simulation with inventory management
- **Flexible Configuration**: Easy configuration for different product categories and scenarios
- **Interactive Visualization**: Tools to analyze and visualize pricing strategies
- **Reward Calibration**: Automated reward shaping for business objectives

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
├── tests/                             # Test files
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
python main.py evaluate --model models/saved/dqn_electronics.pt --episodes 10

# Run a hyperparameter optimization
python main.py tune --agent ppo --config grocery --trials 50
```

### Using the Notebooks

The `notebooks/` directory contains Jupyter notebooks for:
- Exploratory data analysis
- Model comparison
- Business insights

```bash
# Launch Jupyter
jupyter notebook notebooks/
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
4. Train an agent with your configuration

## License

MIT

## Acknowledgments

- The demand model is inspired by economic pricing models and elasticity research
- The RL implementations are based on best practices from the deep RL literature 