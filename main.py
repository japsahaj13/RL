#!/usr/bin/env python3
"""
Main entry point for the MSME Pricing RL project.
"""

import argparse
import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from environments.config import MSMEConfig

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data/processed',
        'models/saved',
        'config',
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)

def train(args):
    """Train an RL agent."""
    print(f"Training agent: {args.agent}")
    print(f"Configuration: {args.config}")
    print(f"Episodes: {args.episodes}")
    
    if args.agent == 'dqn':
        from experiments.train import train_dqn
        train_dqn(
            config_name=args.config,
            num_episodes=args.episodes,
            save_path=args.output
        )
    elif args.agent == 'a2c':
        from experiments.train import train_a2c
        train_a2c(
            config_name=args.config,
            num_episodes=args.episodes,
            save_path=args.output
        )
    elif args.agent == 'ppo':
        from experiments.train import train_ppo
        train_ppo(
            config_name=args.config,
            num_episodes=args.episodes,
            save_path=args.output
        )
    else:
        print(f"Unknown agent: {args.agent}")
        sys.exit(1)

def evaluate(args):
    """Evaluate a trained agent."""
    print(f"Evaluating model: {args.model}")
    print(f"Episodes: {args.episodes}")
    
    from experiments.evaluate import evaluate_model
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render
    )

def tune(args):
    """Run hyperparameter tuning."""
    print(f"Tuning agent: {args.agent}")
    print(f"Configuration: {args.config}")
    print(f"Trials: {args.trials}")
    
    from experiments.hyperparameter_tuning import tune_hyperparameters
    tune_hyperparameters(
        agent_type=args.agent,
        config_name=args.config,
        num_trials=args.trials
    )

def demo(args):
    """Run a demo visualization."""
    print(f"Running demo with model: {args.model}")
    
    from experiments.evaluate import run_demo
    run_demo(
        model_path=args.model,
        save_video=args.save_video
    )

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
        from environments.config import create_electronics_config
        return create_electronics_config()
    elif config_name == 'groceries':
        from environments.config import create_groceries_config
        return create_groceries_config()
    elif config_name == 'toys':
        from environments.config import create_toys_config
        return create_toys_config()
    elif config_name == 'furniture':
        from environments.config import create_furniture_config
        return create_furniture_config()
    elif config_name == 'clothing':
        from environments.config import create_clothing_config
        return create_clothing_config()
    
    # Try to load from YAML file
    config_path = f"config/{config_name}_config.yaml"
    if os.path.exists(config_path):
        config = MSMEConfig(config_path=config_path)
        return config
    
    # Fallback to default config
    print(f"Configuration '{config_name}' not found. Using default configuration.")
    return MSMEConfig()

def generate_configs(args):
    """Generate default configuration files."""
    print("Generating category-specific configuration files...")
    
    from environments.config import (
        create_default_config,
        create_electronics_config,
        create_groceries_config,
        create_toys_config,
        create_furniture_config,
        create_clothing_config
    )
    
    config_dir = os.path.join(PROJECT_ROOT, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    create_default_config().save_to_yaml(os.path.join(config_dir, 'default_config.yaml'))
    create_electronics_config().save_to_yaml(os.path.join(config_dir, 'electronics_config.yaml'))
    create_groceries_config().save_to_yaml(os.path.join(config_dir, 'groceries_config.yaml'))
    create_toys_config().save_to_yaml(os.path.join(config_dir, 'toys_config.yaml'))
    create_furniture_config().save_to_yaml(os.path.join(config_dir, 'furniture_config.yaml'))
    create_clothing_config().save_to_yaml(os.path.join(config_dir, 'clothing_config.yaml'))
    
    print("Category-specific configuration files generated in config/")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MSME Pricing RL Framework")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument('--agent', type=str, default='dqn', 
                             choices=['dqn', 'a2c', 'ppo'], 
                             help='RL algorithm to use')
    train_parser.add_argument('--config', type=str, default='default',
                             help='Configuration name to use')
    train_parser.add_argument('--episodes', type=int, default=300,
                             help='Number of training episodes')
    train_parser.add_argument('--output', type=str, default=None,
                             help='Output path for the trained model')
    
    # Evaluate parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to the trained model')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--render', action='store_true',
                           help='Render the evaluation')
    
    # Tune parser
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning')
    tune_parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'a2c', 'ppo'],
                            help='RL algorithm to tune')
    tune_parser.add_argument('--config', type=str, default='default',
                            help='Configuration name to use')
    tune_parser.add_argument('--trials', type=int, default=50,
                            help='Number of tuning trials')
    
    # Demo parser
    demo_parser = subparsers.add_parser('demo', help='Run a demo visualization')
    demo_parser.add_argument('--model', type=str, required=True,
                           help='Path to the trained model')
    demo_parser.add_argument('--save-video', action='store_true',
                          help='Save a video of the demo')
    
    # Config generator parser
    config_parser = subparsers.add_parser('generate-configs', 
                                         help='Generate default configuration files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Run the specified command
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'tune':
        tune(args)
    elif args.command == 'demo':
        demo(args)
    elif args.command == 'generate-configs':
        generate_configs(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 