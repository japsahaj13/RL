"""
DQN Agent implementation for the MSME Pricing environment.
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union, Any
from collections import deque

from models.networks import DQNPricingNetwork
from utilities.replay_buffer import ReplayMemory, Transition, get_batch_tensor
from environments.config import MSMEConfig
from environments.msme_env import MSMEEnvironment

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MSMEPricingAgent:
    """
    Deep Q-Network agent for MSME pricing decisions.
    
    This agent uses a DQN with experience replay to learn optimal pricing strategies.
    """
    
    def __init__(
            self,
            config: Optional[MSMEConfig] = None,
            env: Optional[MSMEEnvironment] = None,
            gamma: float = 0.95,            # Discount factor
            epsilon_start: float = 0.9,     # Starting exploration rate
            epsilon_end: float = 0.05,      # Final exploration rate
            epsilon_decay: float = 200,     # Exploration decay (in episodes)
            learning_rate: float = 0.001,   # Learning rate
            batch_size: int = 64,           # Batch size for training
            hidden_size: int = 128,         # Hidden layer size
            target_update: int = 10,        # How often to update target network
            memory_size: int = 10000,       # Replay buffer size
            use_double_dqn: bool = True,    # Enable Double DQN
            use_dueling: bool = True        # Enable Dueling network architecture
    ):
        """
        Initialize the DQN pricing agent.
        
        Args:
            config: Configuration object with environment parameters
            env: MSME pricing environment
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay (episodes)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            hidden_size: Size of hidden layers in networks
            target_update: Frequency of target network updates
            memory_size: Size of replay memory buffer
            use_double_dqn: Whether to use Double DQN algorithm
            use_dueling: Whether to use Dueling network architecture
        """
        # Initialize environment if not provided
        if config is not None and env is None:
            self.config = config
            self.env = MSMEEnvironment(config)
        elif env is not None:
            self.env = env
            self.config = env.config
        else:
            # Default configuration
            self.config = MSMEConfig()
            self.env = MSMEEnvironment(self.config)
        
        # Set learning parameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double_dqn = use_double_dqn
        
        # Get state and action sizes from environment
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Initialize networks
        self.policy_net = DQNPricingNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=hidden_size,
            dueling=use_dueling
        ).to(device)
        
        self.target_net = DQNPricingNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=hidden_size,
            dueling=use_dueling
        ).to(device)
        
        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.running_loss = 0.0
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            evaluate: Whether this is for evaluation (no exploration)
            
        Returns:
            Selected action index
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Use greedy policy for evaluation
        if evaluate:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        
        # Decay epsilon
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.episodes_done / self.epsilon_decay)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Exploit: choose best action
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            # Explore: choose random action
            return random.randrange(self.action_size)
    
    def optimize_model(self) -> float:
        """
        Perform one step of optimization on the DQN.
        
        Returns:
            Loss value
        """
        # Check if we have enough samples
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        transitions = self.memory.sample(self.batch_size)
        
        # Transpose batch for easier access
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors and move to device
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float).view(-1, 1).to(device)
        
        # Create mask for non-terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        
        # Only include non-terminal next states
        non_final_next_states = torch.cat([
            s.unsqueeze(0) for s in batch.next_state if s is not None
        ]).to(device)
        
        # Get Q-values for current states and actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Initialize next state values to zero
        next_state_values = torch.zeros(self.batch_size, 1, device=device)
        
        if self.use_double_dqn:
            # Double DQN: use policy net to select actions, target net to evaluate them
            with torch.no_grad():
                # Get actions from policy network
                policy_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
                # Evaluate actions using target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, policy_actions)
        else:
            # Regular DQN: use max value from target network
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
        
        # Calculate expected Q-values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Calculate loss (Huber loss for stability)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(
            self, 
            num_episodes: int = 300, 
            print_every: int = 10,
            visualization_dir: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            print_every: How often to print progress
            visualization_dir: Directory to save visualizations
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Training DQN agent for {num_episodes} episodes...")
        print(f"Using device: {device}")
        
        # Create visualization directory if needed
        if visualization_dir is not None:
            os.makedirs(visualization_dir, exist_ok=True)
        
        # Store episode information
        episode_info = {
            'rewards': [],
            'lengths': [],
            'avg_losses': [],
            'final_inventories': [],
            'profits': [],
            'sales': []
        }
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_loss = 0
            step = 0
            
            # Information for this episode
            episode_prices = []
            episode_comp_prices = []
            episode_demands = []
            episode_sales = []
            episode_inventories = []
            episode_profits = []
            
            while not done:
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition in replay memory
                self.memory.push(
                    torch.FloatTensor(state),
                    action,
                    torch.FloatTensor(next_state) if not done else None,
                    reward,
                    done
                )
                
                # Move to next state
                state = next_state
                
                # Track reward and steps
                episode_reward += reward
                step += 1
                
                # Store information for visualization
                episode_prices.append(info['price'])
                episode_comp_prices.append(info['comp_price'])
                episode_demands.append(info['demand'])
                episode_sales.append(info['sales'])
                episode_inventories.append(info['inventory'])
                episode_profits.append(info['profit'])
                
                # Optimize model
                loss = self.optimize_model()
                episode_loss += loss
                
                # Update the target network every target_update episodes
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                self.steps_done += 1
            
            # Update counters and metrics
            self.episodes_done += 1
            avg_loss = episode_loss / step if step > 0 else 0
            
            # Store episode information
            episode_info['rewards'].append(episode_reward)
            episode_info['lengths'].append(step)
            episode_info['avg_losses'].append(avg_loss)
            episode_info['final_inventories'].append(info['inventory'])
            episode_info['profits'].append(sum(episode_profits))
            episode_info['sales'].append(sum(episode_sales))
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(episode_info['rewards'][-print_every:])
                avg_length = np.mean(episode_info['lengths'][-print_every:])
                avg_loss = np.mean(episode_info['avg_losses'][-print_every:])
                
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Avg Length: {avg_length:.2f}")
                print(f"Avg Loss: {avg_loss:.6f}")
                print(f"Epsilon: {self.epsilon:.2f}")
                print(f"Total Profit: {sum(episode_profits):.2f}")
                print("-" * 40)
            
            # Visualize results periodically
            if visualization_dir is not None:
                # Ensure we save at least 5 visualizations or every episode if fewer than 5 total
                if num_episodes <= 5 or (episode + 1) % max(1, num_episodes // 5) == 0:
                    self.visualize_results({
                        'episode': episode + 1,
                        'prices': episode_prices,
                        'comp_prices': episode_comp_prices,
                        'demands': episode_demands,
                        'sales': episode_sales,
                        'inventories': episode_inventories,
                        'profits': episode_profits
                    }, save_path=os.path.join(visualization_dir, f"episode_{episode+1}.png"))
        
        print("Training complete!")
        
        # Plot training progress
        if visualization_dir is not None:
            self.plot_training_progress(save_path=os.path.join(visualization_dir, "training_progress.png"))
        
        return episode_info
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        # Switch to evaluation mode
        self.policy_net.eval()
        
        total_reward = 0
        total_profit = 0
        total_sales = 0
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_profit = 0
            episode_sales = 0
            step = 0
            
            while not done:
                # Select action (no exploration)
                action = self.select_action(state, evaluate=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Render if requested
                if render:
                    self.env.render()
                
                # Update metrics
                episode_reward += reward
                episode_profit += info['profit']
                episode_sales += info['sales']
                step += 1
                
                # Move to next state
                state = next_state
            
            # Update totals
            total_reward += episode_reward
            total_profit += episode_profit
            total_sales += episode_sales
            episode_lengths.append(step)
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Profit = {episode_profit:.2f}")
        
        # Calculate averages
        avg_reward = total_reward / num_episodes
        avg_profit = total_profit / num_episodes
        avg_sales = total_sales / num_episodes
        avg_length = sum(episode_lengths) / num_episodes
        
        print(f"Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Profit: {avg_profit:.2f}")
        print(f"Average Sales: {avg_sales:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")
        
        # Switch back to training mode
        self.policy_net.train()
        
        return {
            'avg_reward': avg_reward,
            'avg_profit': avg_profit,
            'avg_sales': avg_sales,
            'avg_length': avg_length
        }
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """
        Plot training progress over episodes.
        
        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot smoothed rewards (rolling average)
        smoothing_window = min(10, len(self.episode_rewards))
        if smoothing_window > 0:
            smoothed_rewards = pd.Series(self.episode_rewards).rolling(smoothing_window).mean()
            plt.plot(smoothed_rewards, color='red')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        # Plot losses
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_losses)
        plt.title('Average Loss per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 4)
        epsilons = [self.epsilon_end + (self.epsilon_start - self.epsilon_end) * 
                   math.exp(-1. * i / self.epsilon_decay) for i in range(len(self.episode_rewards))]
        plt.plot(epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training progress plot saved to {save_path}")
        
        plt.close()
    
    def visualize_results(self, episode_info: Dict, save_path: Optional[str] = None):
        """
        Visualize the results of a single episode.
        
        Args:
            episode_info: Dictionary with episode metrics
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 12))
        
        # Plot prices
        plt.subplot(3, 2, 1)
        plt.plot(episode_info['prices'], label='Our Price')
        plt.plot(episode_info['comp_prices'], label='Competitor Price', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title('Price Dynamics')
        plt.legend()
        plt.grid(True)
        
        # Plot demand and sales
        plt.subplot(3, 2, 2)
        plt.plot(episode_info['demands'], label='Demand')
        plt.plot(episode_info['sales'], label='Sales')
        plt.xlabel('Time Step')
        plt.ylabel('Units')
        plt.title('Demand and Sales')
        plt.legend()
        plt.grid(True)
        
        # Plot inventory
        plt.subplot(3, 2, 3)
        plt.plot(episode_info['inventories'])
        plt.axhline(y=self.config.restock_level, color='red', linestyle='--', label='Restock Level')
        plt.xlabel('Time Step')
        plt.ylabel('Units')
        plt.title('Inventory')
        plt.legend()
        plt.grid(True)
        
        # Plot profit per time step
        plt.subplot(3, 2, 4)
        plt.plot(episode_info['profits'])
        plt.xlabel('Time Step')
        plt.ylabel('Profit')
        plt.title('Profit per Time Step')
        plt.grid(True)
        
        # Plot cumulative profit
        plt.subplot(3, 2, 5)
        cumulative_profit = np.cumsum(episode_info['profits'])
        plt.plot(cumulative_profit)
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Profit')
        plt.title('Cumulative Profit')
        plt.grid(True)
        
        # Plot market share
        plt.subplot(3, 2, 6)
        market_share = [s / (d + 1e-6) for s, d in zip(episode_info['sales'], episode_info['demands'])]
        plt.plot(market_share)
        plt.xlabel('Time Step')
        plt.ylabel('Market Share')
        plt.title('Market Share')
        plt.grid(True)
        
        plt.suptitle(f"Episode {episode_info['episode']} Results", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path)
            print(f"Episode visualization saved to {save_path}")
        
        plt.close()
    
    def save(self, path: str) -> None:
        """
        Save the agent's model and parameters.
        
        Args:
            path: Path to save the agent
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the policy network
        self.policy_net.save(f"{path}_policy.pt")
        
        # Save other parameters
        torch.save({
            'epsilon': self.epsilon,
            'episodes_done': self.episodes_done,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses
        }, f"{path}_params.pt")
        
        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's model and parameters.
        
        Args:
            path: Path to the saved agent
        """
        # Load policy network
        self.policy_net = DQNPricingNetwork.load(f"{path}_policy.pt", device)
        
        # Load target network (same weights as policy network)
        self.target_net = DQNPricingNetwork.load(f"{path}_policy.pt", device)
        self.target_net.eval()
        
        # Load other parameters
        params = torch.load(f"{path}_params.pt")
        self.epsilon = params['epsilon']
        self.episodes_done = params['episodes_done']
        self.steps_done = params['steps_done']
        self.episode_rewards = params['episode_rewards']
        self.episode_lengths = params['episode_lengths']
        self.episode_losses = params['episode_losses']
        
        print(f"Agent loaded from {path}") 