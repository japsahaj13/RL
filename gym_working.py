#!/usr/bin/env python
# coding: utf-8

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import gym
from gym import spaces
from gym.envs.registration import register
import os
from datetime import datetime

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================== MSME Configuration and Environment ==================
class MSMEConfig:
    def __init__(
            self, 
            product_name="Sample Product",
            unit_cost=50,             # Cost of producing one unit
            price_tiers=None,         # Available price points
            base_demand=200,          # Base demand at zero price
            price_sensitivity=1.0,    # How much demand drops per unit price
            competitor_effect=0.5,    # Effect of competitor prices
            promo_boost=25,           # Additional demand during promotions
            seasonal_amplitude=15,    # Amplitude of seasonal variations
            holding_cost=0.05,        # Cost per unit of inventory per period
            initial_inventory=150,    # Starting inventory
            restock_level=30,         # Reorder when inventory drops below this
            restock_amount=120,       # Amount to restock
            restock_period=7,
            stockout_penalty=5.0      # Penalty for stockouts
    ):
        self.product_name = product_name
        self.unit_cost = unit_cost
        
        # Calculate price tiers as percentage changes from unit cost
        if price_tiers is None:
            # Define percentage changes
            # A = {-50%, -30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +30%}
            pct_changes = [
                -0.5, -0.3, -0.2, -0.1, 
                -0.05, 0.0, 0.05, 0.1, 
                0.2, 0.3
            ]
            percentage_changes = np.array(pct_changes)
            # Calculate price tiers as price = unit_cost * (1 + pct)
            self.price_tiers = np.array([
                unit_cost * (1 + pct) for pct in percentage_changes
            ])
        else:
            self.price_tiers = price_tiers
            
        self.base_demand = base_demand
        self.price_sensitivity = price_sensitivity
        self.competitor_effect = competitor_effect
        self.promo_boost = promo_boost
        self.seasonal_amplitude = seasonal_amplitude
        self.holding_cost = holding_cost
        self.initial_inventory = initial_inventory
        self.restock_level = restock_level
        self.restock_amount = restock_amount
        self.restock_period = restock_period
        self.stockout_penalty = stockout_penalty
        
    def __str__(self):
        return (f"Product: {self.product_name}\n"
                f"Unit Cost: ${self.unit_cost}\n"
                f"Price Tiers: {self.price_tiers}\n"
                f"Initial Inventory: {self.initial_inventory} units")


class MSMEEnvironment(gym.Env):
    """
    Retail pricing environment for MSME merchants.
    """
    
    def __init__(
            self, config, time_horizon=30, 
            alpha=0.2, beta=0.5, gamma=0.2, delta=0.2
    ):
        """
        Initialize the environment.
        
        Args:
            config: Configuration for the environment
            time_horizon: Number of time steps in an episode
            alpha: Weight for holding cost penalty
            beta: Weight for stockout penalty (increased from 0.2 to 0.5)
            gamma: Weight for price stability penalty
            delta: Weight for excessive discount penalty
        """
        self.config = config
        self.time_horizon = time_horizon
        self.time_step = 0
        
        # Reward shaping parameters
        self.alpha = alpha  # Holding cost weight
        # Stockout penalty weight - increased to prioritize meeting demand
        self.beta = beta    
        self.gamma = gamma  # Price stability penalty weight
        self.delta = delta  # Excessive discount penalty weight
        
        # Action space: price tiers
        self.action_space = spaces.Discrete(len(config.price_tiers))
        
        # Observation space
        # [current_price, price_tier_min, price_tier_max, last_10_own_prices, 
        # last_10_competitor_prices, current_inventory, promo_flag, backlog]
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0] + 
                [0] * 10 + 
                [0] * 10 + 
                [0, 0, 0]
            ),
            high=np.array(
                [np.inf, np.inf, np.inf] + 
                [np.inf] * 10 + 
                [np.inf] * 10 + 
                [np.inf, 1, np.inf]
            ),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        self.time_step = 0
        self.inventory = self.config.initial_inventory
        self.price_history = [0]  # Start with a placeholder price
        self.comp_price_history = [0]
        self.demand_history = []
        self.sales_history = []
        self.inventory_history = [self.inventory]
        self.promotion_history = []
        self.profit_history = []
        self.backlog = 0
        self.stockout_history = []
        
        # Return observation and empty info dict
        return self._get_state(), {}
    
    def _get_state(self):
        """Get the current state representation"""
        # Basic price history (current and previous)
        state = [self.price_history[0]]
        if len(self.price_history) > 1:
            state.append(self.price_history[1])
        else:
            state.append(0)
            
        # Add inventory level
        state.append(self.inventory)
        
        # Add competitor price
        state.append(self.comp_price_history[0])
        
        # Add day of week (0-6)
        state.append(self.time_step % 7)
        
        # Add promotion flag
        promo_flag = 1 if self._is_promotion_period() else 0
        state.append(promo_flag)
        
        # Add backlog
        state.append(self.backlog)
        
        # Add one-hot encoding for time step within the time horizon
        time_encoding = np.zeros(self.time_horizon)
        if self.time_step < self.time_horizon:
            time_encoding[self.time_step] = 1
        state.extend(time_encoding.tolist())
        
        # Convert to numpy array
        state = np.array(state, dtype=np.float32)
        
        # Normalize state values
        return self._normalize_state(state)
    
    def _normalize_state(self, state):
        """Normalize state values to improve training stability"""
        # Extract components needing normalization
        price = state[0]
        prev_price = state[1]
        inventory = state[2]
        comp_price = state[3]
        backlog = state[6]
        
        # Normalize price values by unit cost
        if self.config.unit_cost > 0:
            # Current price relative to unit cost
            state[0] = price / self.config.unit_cost
            # Previous price relative to unit cost  
            state[1] = prev_price / self.config.unit_cost
            # Competitor price relative to unit cost
            state[3] = comp_price / self.config.unit_cost
        
        # Normalize inventory by initial inventory
        if self.config.initial_inventory > 0:
            state[2] = inventory / self.config.initial_inventory
        
        # Normalize backlog by initial inventory as reference
        if self.config.initial_inventory > 0:
            state[6] = backlog / self.config.initial_inventory
        
        return state
    
    def _demand_model(self, price, comp_price, promo_flag, time_step):
        """MSME-realistic demand model with price, competition, seasonality and promotions"""
        # Base demand reduction based on price
        base_effect = self.config.base_demand - self.config.price_sensitivity * price
        
        # Competitor effect - gain demand if our price is lower
        comp_effect = max(0, comp_price - price) * self.config.competitor_effect
        
        # Seasonal effect - weekly pattern
        seasonal = self.config.seasonal_amplitude * np.sin(2 * np.pi * time_step / 7)
        
        # Promotion boost
        promo_boost = self.config.promo_boost if promo_flag else 0
        
        # Final demand calculation with floor at zero
        demand = max(0, base_effect + comp_effect + seasonal + promo_boost)
        
        return max(0, int(demand))
    
    def _is_promotion_period(self):
        """Check if current period is a promotion period"""
        return self.time_step % 14 == 0  # Promotion every 2 weeks
    
    def _sample_competitor_price(self, our_price):
        """
        Generate a competitor price based on our price and market conditions.
        Competitors are more likely to undercut when inventory levels are low
        and more likely to raise prices when inventory is high.
        """
        # Base range for competitor price
        min_price = our_price * 0.75
        max_price = our_price * 1.25
        
        # Adjust competitor behavior based on inventory level
        inventory_ratio = self.inventory / self.config.initial_inventory
        
        # If our inventory is low, competitors might be more aggressive
        if inventory_ratio < 0.3:
            # Competitor more likely to undercut when we have low inventory
            min_price = our_price * 0.7
            max_price = our_price * 1.1
        # If our inventory is high, competitors might be less aggressive
        elif inventory_ratio > 0.8:
            # Competitor likely to raise prices when we have high inventory
            min_price = our_price * 0.9
            max_price = our_price * 1.3
        
        # Random price within adjusted range
        return np.random.uniform(min_price, max_price)
    
    def _restock_inventory(self):
        """Restock inventory if needed - with dynamic restock amount based on demand patterns"""
        
        # Check if inventory is below restock level
        if self.inventory < self.config.restock_level:
            # Calculate average demand over the past few periods (if available)
            recent_demand = self.demand_history[-min(5, len(self.demand_history)):]
            if recent_demand:
                # Base restock on recent demand patterns plus a buffer
                avg_demand = sum(recent_demand) / len(recent_demand)
                # Restock enough for next 5 periods based on recent demand, plus buffer
                restock_qty = int(max(
                    self.config.restock_amount, 
                    avg_demand * 5 * 1.2
                ))
            else:
                # Use default restock amount if no demand history
                restock_qty = self.config.restock_amount
                
            self.inventory += restock_qty
            return restock_qty
        else:
            return 0

    def step(self, action):
        """Take a step in the environment based on the selected action"""

        # Convert action to price
        price = self.config.price_tiers[action]

        # Previous price
        prev_price = self.price_history[0] if self.price_history else price

        # Update price history
        self.price_history.insert(0, price)
        if len(self.price_history) > 10:
            self.price_history.pop()

        # Generate competitor price
        comp_price = self._sample_competitor_price(price)
        self.comp_price_history.insert(0, comp_price)
        if len(self.comp_price_history) > 10:
            self.comp_price_history.pop()

        # Promotion check
        promo_flag = self._is_promotion_period()
        self.promotion_history.append(promo_flag)

        # Calculate demand
        predicted_demand = self._demand_model(price, comp_price, promo_flag, self.time_step)
        self.demand_history.append(predicted_demand)

        # Actual sales (limited by inventory)
        sales = min(predicted_demand, self.inventory)
        self.sales_history.append(sales)

        # Unfulfilled demand
        unfulfilled = max(0, predicted_demand - sales)
        self.backlog = unfulfilled
        self.stockout_history.append(unfulfilled)

        # Update inventory
        self.inventory -= sales

        # Restock if needed
        restock_qty = self._restock_inventory()

        # Calculate revenue and costs
        revenue = sales * price
        cost = sales * self.config.unit_cost
        holding_cost = self.inventory * self.config.holding_cost
        stockout_penalty = unfulfilled * self.config.stockout_penalty
        
        # Calculate discount penalty
        discount_penalty = max(0, self.config.unit_cost - price)
        
        # Calculate fill rate (sales as percentage of demand) - to reward meeting more demand
        fill_rate_bonus = 0
        if predicted_demand > 0:
            fill_rate = sales / predicted_demand
            # Bonus increases as fill rate approaches 100%
            fill_rate_bonus = fill_rate * 100  # Stronger bonus for higher fill rates

        # Calculate revenue minus cost (base profit component)
        base_revenue_cost = revenue - cost
        
        # Calculate shaped reward using tunable weights
        shaped_reward = (
            base_revenue_cost
            - self.alpha * holding_cost
            - self.beta * stockout_penalty
            - self.gamma * abs(price - prev_price)
            + fill_rate_bonus  # Bonus for meeting demand
        )
        
        # Only apply discount penalty when price < 80% of unit cost
        unit_cost_threshold = 0.8 * self.config.unit_cost
        if price < unit_cost_threshold:
            shaped_reward -= self.delta * discount_penalty
        
        # Calculate base profit (for analysis/tracking purposes)
        base_profit = base_revenue_cost - holding_cost - stockout_penalty
        
        # Store base profit for analysis
        self.profit_history.append(base_profit)
        self.inventory_history.append(self.inventory)

        # Advance time
        self.time_step += 1

        # Done check (terminated)
        terminated = self.time_step >= self.time_horizon
        # No truncation in this environment
        truncated = False

        # Get next state
        next_state = self._get_state()

        # Info summary - include all reward components for analysis
        info = {
            'price': price,
            'comp_price': comp_price,
            'demand': predicted_demand,
            'sales': sales,
            'unfulfilled': unfulfilled,
            'inventory': self.inventory,
            'restocked': restock_qty,
            'profit': base_profit,
            'shaped_reward': shaped_reward,
            'promo_active': promo_flag,
            'holding_cost': holding_cost,
            'stockout_penalty': stockout_penalty,
            'price_stability_penalty': self.gamma * abs(price - prev_price),
            'discount_penalty': self.delta * discount_penalty,
            'fill_rate_bonus': fill_rate_bonus,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta
        }

        return next_state, shaped_reward, terminated, truncated, info

# ================== Reinforcement Learning Components ==================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class DQNPricingNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, dueling=True):
        super(DQNPricingNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        if dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )
        else:
            # Traditional Q-network output layer
            self.output_layer = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        if self.dueling:
            values = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine value and advantage streams 
            # (subtracting mean advantage to ensure identifiability)
            q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.output_layer(features)
        
        return q_values

class MSMEPricingAgent:
    def __init__(
            self,
            config=None,
            env=None,
            gamma=0.95,            # Discount factor
            epsilon_start=0.9,     # Starting exploration rate
            epsilon_end=0.05,      # Final exploration rate
            epsilon_decay=200,     # Exploration decay (in episodes)
            learning_rate=0.001,   # Learning rate
            batch_size=64,         # Batch size for training
            hidden_size=128,       # Hidden layer size
            target_update=10,      # How often to update target network
            memory_size=10000,     # Replay buffer size
            use_double_dqn=True,   # Enable Double DQN
            use_dueling=True       # Enable Dueling network architecture
    ):
        self.config = config
        self.env = env
        
        # RL parameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # Now in episodes, not steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        
        # Get state and action dimensions
        state_tuple = env.reset()
        # Handle both formats: just state or (state, info) tuple
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        self.state_size = len(state)
        self.action_size = len(config.price_tiers)
        
        # Initialize networks and optimizer
        self.policy_net = DQNPricingNetwork(
            self.state_size, 
            self.action_size, 
            hidden_size,
            dueling=use_dueling
        ).to(device)
        
        self.target_net = DQNPricingNetwork(
            self.state_size, 
            self.action_size, 
            hidden_size,
            dueling=use_dueling
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Counters and trackers
        self.steps_done = 0
        self.episodes_done = 0
        
        # Tracking variables for visualization
        self.episode_rewards = []
        self.avg_losses = []
        self.epsilon_values = []
        
        # Q-value tracking for diagnostics
        self.q_value_history = []
    
    def select_action(self, state, evaluate=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluate: If True, use greedy policy (epsilon=0)
        
        Returns:
            Selected action
        """
        # Always use greedy policy during evaluation
        if evaluate:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                
                # Store Q-values for diagnostics
                if len(self.q_value_history) < 1000:  # Limit storage to prevent memory issues
                    self.q_value_history.append(q_values.max().item())
                    
                return q_values.max(1)[1].item()
        
        # During training, use epsilon-greedy
        sample = random.random()
        
        # Calculate epsilon based on episodes, not steps
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.episodes_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            # Exploitation: Choose best action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                
                # Store Q-values for diagnostics
                if len(self.q_value_history) < 1000:  # Limit storage
                    self.q_value_history.append(q_values.max().item())
                    
                return q_values.max(1)[1].item()
        else:
            # Exploration: Choose random action
            return random.randrange(self.action_size)
    
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create mask for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                             batch.next_state)), 
                                   device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) 
                                         for s in batch.next_state if s is not None]).to(device)
        
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)
        
        # Scale rewards for numerical stability (divide by 100 to keep in reasonable range)
        # This prevents exploding gradients while preserving relative reward values
        reward_scale_factor = 100.0
        scaled_reward_batch = reward_batch / reward_scale_factor
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN implementation:
        # 1. Get actions that would be selected by policy_net for next states
        # 2. Get target Q-values from target_net for those actions
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_mask.sum() > 0:  # Check if there are any non-final states
            # Get actions from policy network (Double DQN)
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            # Get values from target network for those actions
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).detach().squeeze()
        
        # Compute expected Q values
        expected_state_action_values = scaled_reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Compute Huber loss (more robust than MSE for outliers)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=300, print_every=10, visualization_dir=None):
        """Train the DQN agent"""
        if visualization_dir is not None and not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
            
        all_rewards = []
        all_losses = []
        
        start_time = datetime.now()
        best_episode_reward = float('-inf')
        best_episode = 0
        
        for episode in range(num_episodes):
            self.episodes_done = episode  # For exploration rate calculation
            
            # Initialize the environment
            state_tuple = self.env.reset()
            # Handle both formats: just state or (state, info) tuple
            state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
            
            # Initialize episode variables
            episode_reward = 0
            episode_losses = []
            
            done = False
            
            while not done:
                # Select and perform an action
                action = self.select_action(state)
                
                # Take the action
                next_state_tuple, reward, terminated, truncated, info = self.env.step(action)
                # Handle both formats: just state or (state, info) tuple
                next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                
                done = terminated or truncated
                episode_reward += reward
                
                # Store transition in memory
                self.memory.push(state, action, next_state if not done else None, reward, float(done))
                
                # Move to next state
                state = next_state
                
                # Perform one step of optimization
                if len(self.memory) >= self.batch_size:
                    loss = self.optimize_model()
                    if loss is not None:
                        episode_losses.append(loss)
                        
            # End of episode processing
            all_rewards.append(episode_reward)
            
            # Store average loss for the episode
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            all_losses.append(avg_loss)
            self.avg_losses.append(avg_loss)
            
            # Track average reward
            self.episode_rewards.append(episode_reward)
            
            # Calculate current epsilon for tracking
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     math.exp(-1. * self.episodes_done / self.epsilon_decay)
            
            # Store epsilon value for visualization
            self.epsilon_values.append((episode, epsilon))
            
            # Track best episode and keep for evaluation
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode = episode
            
            # Print progress
            if episode % print_every == 0:
                avg_reward = np.mean(all_rewards[-print_every:]) if len(all_rewards) >= print_every else np.mean(all_rewards)
                print(f"Episode {episode}/{num_episodes} | Avg Reward: ${avg_reward:.2f} | Epsilon: {epsilon:.2f}")
                
                # Evaluate the agent periodically
                if visualization_dir is not None:
                    # Generate evaluation visualization
                    total_reward, episode_info = self.evaluate()
                    
                    # Generate and save visualization instead of displaying
                    viz_filename = f"{visualization_dir}/eval_summary_ep{episode}.png"
                    self.visualize_results(episode_info, save_path=viz_filename)
                    
                    # Plot training progress every 50 episodes and save to file
                    if episode % 50 == 0 and episode > 0:
                        progress_filename = f"{visualization_dir}/training_progress_ep{episode}.png"
                        self.plot_training_progress(save_path=progress_filename)
        
        print(f"Training complete! Best episode reward: ${best_episode_reward:.2f}")
        
        # Print Q-value statistics if available
        if self.q_value_history:
            q_values = np.array(self.q_value_history)
            print(f"Q-value statistics: min={q_values.min():.2f}, max={q_values.max():.2f}, mean={q_values.mean():.2f}")
        
        return all_rewards, all_losses
    
    def evaluate(self):
        """Evaluate the trained agent on the environment"""
        # Reset the environment
        state_tuple = self.env.reset()
        # Handle both formats: just state or (state, info) tuple
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        done = False
        total_reward = 0
        info_history = []
        
        while not done:
            # Select action using policy network (no exploration)
            action = self.select_action(state, evaluate=True)
            
            # Take action
            next_state_tuple, reward, terminated, truncated, info = self.env.step(action)
            # Handle both formats: just state or (state, info) tuple
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            done = terminated or truncated
            total_reward += reward
            
            # Store info for analysis
            info_history.append(info)
            
            # Move to next state
            state = next_state
            
        return total_reward, info_history
    
    def plot_training_progress(self, save_path=None):
        """Visualize training progress with multiple plots"""
        plt.figure(figsize=(12, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        rewards_smoothed = pd.Series(self.episode_rewards).rolling(window=10).mean()
        plt.plot(self.episode_rewards, alpha=0.3)
        plt.plot(rewards_smoothed)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot training loss
        plt.subplot(2, 2, 2)
        if self.avg_losses:
            losses_smoothed = pd.Series(self.avg_losses).rolling(window=10).mean()
            plt.plot(self.avg_losses, alpha=0.3)
            plt.plot(losses_smoothed)
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
        
        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        if self.epsilon_values:
            episodes, epsilons = zip(*self.epsilon_values)
            plt.plot(episodes, epsilons)
            plt.title('Exploration Rate (Epsilon)')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
        
        # Plot Q-values over time
        plt.subplot(2, 2, 4)
        if self.q_value_history:
            plt.plot(self.q_value_history)
            plt.title('Q-Value Evolution')
            plt.xlabel('Step')
            plt.ylabel('Max Q-Value')
        
        plt.tight_layout()
        
        # Save figure if a path is provided, otherwise save with default name
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Training progress plot saved to {save_path}")
        else:
            # Create a default filename
            filename = f"training_progress_{self.config.product_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Training progress plot saved to {filename}")
    
    def visualize_results(self, episode_info, save_path=None):
        """Generate comprehensive visualization of an episode"""
        # Extract data from episode info
        # Create time_steps if not present in the info dictionary
        if 'time_step' in episode_info[0]:
            time_steps = [info['time_step'] for info in episode_info]
        else:
            # Use sequential steps if time_step is not available
            time_steps = list(range(len(episode_info)))
            
        prices = [info['price'] for info in episode_info]
        comp_prices = [info['comp_price'] for info in episode_info]
        demands = [info['demand'] for info in episode_info]
        sales = [info['sales'] for info in episode_info]
        inventory = [info['inventory'] for info in episode_info]
        profits = [info['profit'] for info in episode_info]
        promos = [info['promo_active'] for info in episode_info]
        
        plt.figure(figsize=(14, 12))
        
        # Price vs. Competitor subplot
        plt.subplot(4, 1, 1)
        plt.plot(time_steps, prices, 'b-', label='Our Price', linewidth=2)
        plt.plot(time_steps, comp_prices, 'r--', label='Competitor Price', linewidth=2)
        # Mark promotion periods
        for i, promo in enumerate(promos):
            if promo:
                plt.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, alpha=0.2, color='green')
        
        plt.title('Pricing Strategy vs. Competitor')
        plt.xlabel('Time Period')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        # Demand vs Sales subplot
        plt.subplot(4, 1, 2)
        plt.plot(time_steps, demands, 'g-', label='Demand', linewidth=2)
        plt.plot(time_steps, sales, 'b-', label='Actual Sales', linewidth=2)
        plt.fill_between(time_steps, sales, demands, color='red', alpha=0.3, label='Unfulfilled')
        plt.title('Demand vs Sales')
        plt.xlabel('Time Period')
        plt.ylabel('Units')
        plt.legend()
        plt.grid(True)
        
        # Inventory subplot
        plt.subplot(4, 1, 3)
        plt.plot(time_steps, inventory, 'b-', linewidth=2)
        plt.axhline(y=self.config.restock_level, color='r', linestyle='--', label=f'Restock Level ({self.config.restock_level})')
        plt.title('Inventory Level')
        plt.xlabel('Time Period')
        plt.ylabel('Units')
        plt.legend()
        plt.grid(True)
        
        # Profit subplot
        plt.subplot(4, 1, 4)
        plt.bar(time_steps, profits, color='green')
        plt.title('Profit per Period')
        plt.xlabel('Time Period')
        plt.ylabel('Profit ($)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Calculate summary statistics
        total_demand = sum(demands)
        total_sales = sum(sales)
        stockout_rate = (total_demand - total_sales) / total_demand if total_demand > 0 else 0
        avg_price = sum(prices) / len(prices)
        avg_comp = sum(comp_prices) / len(comp_prices)
        total_profit = sum(profits)
        
        print("\n===== PERFORMANCE SUMMARY =====")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Average Price: ${avg_price:.2f} (Competitor: ${avg_comp:.2f})")
        print(f"Total Demand: {total_demand} units")
        print(f"Actual Sales: {total_sales} units")
        print(f"Stockout Rate: {stockout_rate*100:.1f}%")
        
        # Save figure if a path is provided, otherwise save with default name
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Episode visualization saved to {save_path}")
        else:
            # Create a default filename
            filename = f"episode_results_{self.config.product_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Episode visualization saved to {filename}")
        
        return prices, demands, sales, profits, inventory
    
    def price_sensitivity_analysis(self, save_path=None):
        """Analyze profitability across different price points"""
        # Reset environment to initial state
        self.env.reset()
        
        # Try each price point for the first period
        results = []
        
        for i, price in enumerate(self.config.price_tiers):
            # Reset environment
            self.env.reset()
            
            # Take step with fixed price
            next_state_tuple, reward, terminated, truncated, info = self.env.step(i)
            
            # Record results
            results.append({
                'price': price,
                'demand': info['demand'],
                'sales': info['sales'],
                'profit': info['profit']
            })
        
        # Plot results
        plt.figure(figsize=(10, 6))
        prices = [r['price'] for r in results]
        profits = [r['profit'] for r in results]
        demands = [r['demand'] for r in results]
        
        # Primary axis for profit
        ax1 = plt.gca()
        ax1.plot(prices, profits, 'b-o', linewidth=2)
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Profit ($)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Secondary axis for demand
        ax2 = ax1.twinx()
        ax2.plot(prices, demands, 'r--o', linewidth=2)
        ax2.set_ylabel('Demand (Units)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Mark cost line
        plt.axvline(x=self.config.unit_cost, color='g', linestyle='--', 
                    label=f'Unit Cost (${self.config.unit_cost})')
        
        plt.title('Price Sensitivity Analysis')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure if a path is provided, otherwise show it
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Price sensitivity analysis saved to {save_path}")
        else:
            # Create a default filename
            filename = f"price_sensitivity_{self.config.product_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Price sensitivity analysis saved to {filename}")
        
        return results
    
    def competitor_analysis(self, save_path=None):
        """Analyze price strategy relative to simulated competitors"""
        # Run simulation with trained policy
        state_tuple = self.env.reset()
        # Handle both formats: just state or (state, info) tuple
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        done = False
        comp_data = []
        
        while not done:
            action = self.select_action(state, evaluate=True)
            next_state_tuple, reward, terminated, truncated, info = self.env.step(action)
            # Handle both formats: just state or (state, info) tuple
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            # Check if episode is done
            done = terminated or truncated
            
            comp_data.append({
                'time_step': self.env.time_step,
                'our_price': info['price'],
                'comp_price': info['comp_price'],
                'price_diff': info['price'] - info['comp_price'],
                'demand': info['demand'],
                'profit': info['profit']
            })
            
            state = next_state
            
        # Calculate price position statistics
        below_comp = sum(1 for d in comp_data if d['price_diff'] < 0)
        above_comp = sum(1 for d in comp_data if d['price_diff'] > 0)
        equal_comp = sum(1 for d in comp_data if d['price_diff'] == 0)
        
        # Plot price position over time
        plt.figure(figsize=(12, 6))
        time_steps = [d['time_step'] for d in comp_data]
        price_diffs = [d['price_diff'] for d in comp_data]
        
        plt.bar(time_steps, price_diffs)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Price Positioning vs Competitors')
        plt.xlabel('Time Period')
        plt.ylabel('Price Difference ($) [Positive = Above Competitor]')
        plt.grid(True)
        
        # Add summary
        summary = f"Price Position: Below Competitor {below_comp}/{len(comp_data)} times, "
        summary += f"Above {above_comp}/{len(comp_data)} times, Equal {equal_comp}/{len(comp_data)} times"
        plt.figtext(0.5, 0.01, summary, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure if a path is provided, otherwise save with default name
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Competitor analysis saved to {save_path}")
        else:
            # Create a default filename
            filename = f"competitor_analysis_{self.config.product_name.replace(' ', '_')}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Competitor analysis saved to {filename}")
        
        return comp_data

# ================== Debugging and Testing Functions ==================
def debug_walkthrough():
    """Step-by-step debug of the environment to verify functionality"""
    # Create a fixed config
    config = create_fixed_default_config()
    
    # Create environment
    env = MSMEEnvironment(config, time_horizon=5)  # Short time horizon for debugging
    
    print("===== MSME MODEL DEBUG WALKTHROUGH =====")
    print(f"Configuration:\n{config}")
    
    # Reset the environment
    state_tuple = env.reset()
    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
    print(f"\nInitial State: {state}\n")
    
    # Try each price tier
    for i, price in enumerate(config.price_tiers):
        print(f"\n\n===== STEP {i+1}: SETTING PRICE TO ${price:.2f} =====")
        
        # Take a step with this price
        next_state_tuple, reward, terminated, truncated, info = env.step(i)
        next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
        done = terminated or truncated
        
        # Show results
        print(f"\n--- Results Summary ---")
        print(f"Reward: ${reward:.2f}")
        print(f"New State: {next_state}")
        
        # Show time progression
        print(f"\nTime step progressed from {env.time_step-1} to {env.time_step}")
        
        # Break if the episode is done
        if done:
            print("\nEpisode finished")
            break
    
    # Final summary
    print("\n===== DEBUG WALKTHROUGH SUMMARY =====")
    print(f"Demand History: {env.demand_history}")
    print(f"Sales History: {env.sales_history}")
    print(f"Inventory History: {env.inventory_history}")
    print(f"Profit History: {env.profit_history}")
    
    return env

def test_msme_pricing():
    """Test a single pricing scenario to debug the model"""
    # Create a test configuration with realistic values
    config = MSMEConfig(
        product_name="Test Product",
        unit_cost=50,                  # Cost per unit
        price_tiers=np.array([75, 85, 100, 115, 130]),  # Price points
        base_demand=200,               # Good base demand
        price_sensitivity=1.5,         # More reasonable price sensitivity
        competitor_effect=0.5,         # Moderate competitor effect
        promo_boost=30,                # Good promotion boost
        seasonal_amplitude=15,         # Moderate seasonality
        holding_cost=0.05,             # Lower holding cost
        initial_inventory=300,         # Higher initial inventory
        restock_level=50,              # Restock level
        restock_amount=200,            # Restock amount
        restock_period=7               # Weekly restocking
    )
    
    # Create environment
    env = MSMEEnvironment(config, time_horizon=10)  # Shorter time horizon for testing
    
    # Test different price points
    print("\n===== TESTING DIFFERENT PRICE POINTS =====")
    
    # Reset environment
    state_tuple = env.reset()
    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
    
    # Try each price point
    for i, price in enumerate(config.price_tiers):
        print(f"\n===== TESTING PRICE TIER {i}: ${price:.2f} =====")
        next_state_tuple, reward, terminated, truncated, info = env.step(i)
        done = terminated or truncated
        print(f"Results: Reward=${reward:.2f}, Done={done}")
    
    # Show final summary
    print("\n===== TEST SUMMARY =====")
    print(f"Demand History: {env.demand_history}")
    print(f"Sales History: {env.sales_history}")
    print(f"Profit History: {env.profit_history}")
    
    return env

# ================== Predefined Configurations ==================
def create_fixed_default_config():
    """Create a more realistic default configuration"""
    return MSMEConfig(
        product_name="Sample MSME Product",
        unit_cost=50,                     # Cost per unit
        # Price tiers are now calculated automatically as percentages of unit cost
        base_demand=200,                  # Higher base demand
        price_sensitivity=1.5,            # Increased price sensitivity for better responsiveness
        competitor_effect=0.8,            # Stronger competitor effect
        promo_boost=25,                   # Good promotion boost
        seasonal_amplitude=15,            # Moderate seasonality
        holding_cost=0.05,                # Lower holding cost (5% of value per period)
        initial_inventory=150,            # Moderate initial inventory
        restock_level=40,                 # Higher restock level to avoid stockouts
        restock_amount=120,               # Moderate restock amount
        stockout_penalty=10.0             # Higher stockout penalty to prioritize meeting demand
    )

def create_fashion_retailer_config():
    """Create a configuration for a fashion retailer"""
    return MSMEConfig(
        product_name="Seasonal Fashion Item",
        unit_cost=30,
        # Price tiers calculated automatically as percentages of unit cost
        base_demand=100,
        price_sensitivity=3.0,
        competitor_effect=0.6,
        promo_boost=40,  # Strong promotion effect
        seasonal_amplitude=25,  # High seasonality
        initial_inventory=200,
        restock_level=40,
        restock_amount=160,
        restock_period=14  # Bi-weekly restocking
    )

def create_grocery_store_config():
    """Create a configuration for a grocery item"""
    return MSMEConfig(
        product_name="Grocery Staple",
        unit_cost=2,
        # Price tiers calculated automatically as percentages of unit cost
        base_demand=500,
        price_sensitivity=150,  # Very price sensitive
        competitor_effect=1.2,  # Strong competitor effect
        promo_boost=200,  # Big promotion boost
        seasonal_amplitude=50,  # Moderate seasonality
        initial_inventory=1000,
        restock_level=200,
        restock_amount=800,
        restock_period=3  # Frequent restocking
    )

def create_electronics_store_config():
    """Create a configuration for an electronics item"""
    return MSMEConfig(
        product_name="Consumer Electronics",
        unit_cost=200,
        # Price tiers calculated automatically as percentages of unit cost
        base_demand=200,
        price_sensitivity=0.5,  # Less price sensitive
        competitor_effect=0.4,
        promo_boost=15,
        seasonal_amplitude=10,
        initial_inventory=80,
        restock_level=15,
        restock_amount=65,
        restock_period=21  # Less frequent restocking
    )

# ================== MSME-Friendly Interface ==================
def create_msme_config():
    """Interactive function to create MSME configuration"""
    print("===== MSME Product Configuration =====")
    product_name = input("Enter product name: ")
    
    try:
        unit_cost = float(input("Production cost per unit ($): "))
        
        # Now instead of asking for price tiers, we'll explain the percentage-based approach
        print("\nThe system uses pricing tiers as percentages of your unit cost.")
        print("By default, these are: -50%, -30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +30%")
        
        # Optional: Allow them to specify custom price modifiers
        custom_pricing = input("Would you like to use custom price modifiers? (y/n): ").lower() == 'y'
        
        price_tiers = None
        if custom_pricing:
            print("Enter price modifiers as decimals (e.g., 0.1 for +10%, -0.2 for -20%)")
            print("Separate each value with a space, for example: -0.3 -0.1 0 0.1 0.3")
            modifiers_input = input("Price modifiers: ")
            try:
                modifiers = [float(m) for m in modifiers_input.split()]
                if modifiers:
                    # Calculate actual price tiers based on unit cost
                    price_tiers = np.array([unit_cost * (1 + m) for m in modifiers])
            except ValueError:
                print("Invalid modifiers. Using default pricing tiers.")
        
        base_demand = float(input("Estimated base demand (units at very low price): "))
        price_sensitivity = float(input("Price sensitivity (units lost per $1 increase): "))
        initial_inventory = int(input("Initial inventory (units): "))
        restock_amount = int(input("Restock quantity (units): "))
        
        config = MSMEConfig(
            product_name=product_name,
            unit_cost=unit_cost,
            price_tiers=price_tiers,  # Will be None to use default or custom if specified
            base_demand=base_demand,
            price_sensitivity=price_sensitivity,
            initial_inventory=initial_inventory,
            restock_amount=restock_amount
        )
        
        print("\nConfiguration created successfully!")
        print(config)
        return config
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

def run_msme_dqn_model():
    """Main function to run the MSME DQN pricing model"""
    print("===== MSME Dynamic Pricing Simulator =====")
    print("This tool helps small businesses optimize pricing with realistic constraints")
    
    # Option to use default config or create custom
    use_default = input("Use default configuration? (y/n): ").lower() == 'y'
    if not use_default:
        config = create_msme_config()
    
    config = None
    if use_default:
        config_type = input(
            "Select configuration type (1=Default, 2=Fashion, 3=Grocery, 4=Electronics): "
        )
        if config_type == '2':
            config = create_fashion_retailer_config()
        elif config_type == '3':
            config = create_grocery_store_config()
        elif config_type == '4':
            config = create_electronics_store_config()
        else:
            config = create_fixed_default_config()
    
    # Return early if config creation failed
    if config is None:
        print("Failed to create configuration. Exiting.")
        return None, None, None
    
    # Option to use reward shaping
    use_shaping = input(
        "Use Bayesian optimization for reward shaping? (y/n): "
    ).lower() == 'y'
    
    # Create environment and agent
    time_horizon = int(
        input("Number of time periods to simulate (default 30): ") or "30"
    )
    
    # Perform Bayesian optimization if requested
    alpha, beta, gamma, delta = 0.0, 0.0, 0.0, 0.0
    if use_shaping:
        try:
            # Bayesian optimization of reward shaping parameters
            print("Running Bayesian optimization...")
            alpha, beta, gamma, delta = calibrate_reward_shaping(
                config=config,
                time_horizon=time_horizon,
                num_episodes=5,
                num_optimization_trials=30
            )
            print(
                f"Using optimized parameters: α={alpha:.4f}, β={beta:.4f}, "
                f"γ={gamma:.4f}, δ={delta:.4f}"
            )
        except Exception as e:
            print(f"Error during optimization: {e}")
            print("Using default parameters instead.")
    else:
        # Use default values
        alpha, beta, gamma, delta = 1.0, 1.0, 0.2, 0.2
        print(
            f"Using default reward parameters: α={alpha}, β={beta}, "
            f"γ={gamma}, δ={delta}"
        )
    
    # Create environment with optimized parameters
    env = MSMEEnvironment(
        config, time_horizon=time_horizon, 
        alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )
    
    # Advanced DQN options
    use_double_dqn = input("Use Double DQN? (y/n, default: y): ").lower() != 'n'
    use_dueling = input(
        "Use Dueling network architecture? (y/n, default: y): "
    ).lower() != 'n'
    
    # Set epsilon decay based on episode count
    num_episodes = int(input("Number of training episodes (default 300): ") or "300")
    epsilon_decay = num_episodes // 3  # Decay over one-third of total episodes
    
    # Create agent with enhanced options
    agent = MSMEPricingAgent(
        config=config,
        env=env,
        gamma=0.95,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=epsilon_decay,
        learning_rate=0.001,
        batch_size=64,
        hidden_size=128,
        target_update=10,
        memory_size=10000,
        use_double_dqn=use_double_dqn,
        use_dueling=use_dueling
    )
    
    # Create a directory for outputs if it doesn't exist
    output_dir = "msme_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    product_name = config.product_name.replace(" ", "_")
    base_filename = f"{output_dir}/{product_name}_{timestamp}"
    
    # Train the agent with output directory for visualizations
    print("\nTraining DQN pricing agent...")
    rewards, losses = agent.train(
        num_episodes=num_episodes, 
        print_every=20, 
        visualization_dir=output_dir
    )
    
    # Save training progress visualization
    agent.plot_training_progress(save_path=f"{base_filename}_training_progress.png")
    
    # Evaluate with full episode
    print("\nEvaluating trained agent...")
    total_reward, episode_info = agent.evaluate()
    print(f"Total profit for evaluation episode: ${total_reward:.2f}")
    
    # Visualize evaluation results
    agent.visualize_results(episode_info, save_path=f"{base_filename}_evaluation.png")
    
    # Price sensitivity analysis
    print("\nAnalyzing price sensitivity...")
    price_analysis = agent.price_sensitivity_analysis(save_path=f"{base_filename}_price_sensitivity.png")
    
    # Competitor analysis
    print("\nAnalyzing competitor pricing strategy...")
    agent.competitor_analysis(save_path=f"{base_filename}_competitor_analysis.png")
    
    # Recommendations
    print("\n===== MSME Pricing Recommendations =====")
    print(f"For {config.product_name}:")
    print("1. Optimal dynamic pricing strategy implemented")
    print(
        f"2. Expected total profit over {time_horizon} periods: ${total_reward:.2f}"
    )
    
    # Find best performing individual price
    best_price = max(price_analysis, key=lambda x: x['profit'])
    print(
        f"3. Best fixed price point: ${best_price['price']:.2f} "
        f"(Est. profit: ${best_price['profit']:.2f})"
    )
    
    # Inventory management recommendations
    stockout_periods = sum(1 for info in episode_info if info['demand'] > info['sales'])
    if stockout_periods > 0:
        print(f"4. Inventory WARNING: Stock-outs occurred in {stockout_periods} periods")
        print(
            f"   Consider increasing restock amount from {config.restock_amount} "
            f"or reducing restock period"
        )
    
    # Promotion effectiveness
    promo_periods = sum(1 for info in episode_info if info['promo_active'])
    if promo_periods > 0:
        promo_profits = sum(
            info['profit'] for info in episode_info if info['promo_active']
        )
        non_promo_profits = sum(
            info['profit'] for info in episode_info if not info['promo_active']
        )
        
        avg_promo_profit = promo_profits / promo_periods if promo_periods > 0 else 0
        avg_non_promo_profit = (
            non_promo_profits / (len(episode_info) - promo_periods) 
            if (len(episode_info) - promo_periods) > 0 else 0
        )
        
        if avg_promo_profit > avg_non_promo_profit:
            print(
                f"5. Promotions are effective: Avg profit ${avg_promo_profit:.2f} "
                f"vs ${avg_non_promo_profit:.2f} in regular periods"
            )
        else:
            print(
                f"5. Promotions may need adjustment: Avg profit ${avg_promo_profit:.2f} "
                f"vs ${avg_non_promo_profit:.2f} in regular periods"
            )
    
    # Reward shaping insights
    if use_shaping:
        print("\nReward Shaping Parameters:")
        print(f"  Holding cost weight (α): ${alpha:.4f}")
        print(f"  Stockout penalty weight (β): ${beta:.4f}")
        print(f"  Price stability weight (γ): ${gamma:.4f}")
        print(f"  Excessive discount weight (δ): ${delta:.4f}")
        print("  Impact: More balanced pricing, inventory management, and price stability")
    
    # DQN enhancements report
    print("\nDQN Enhancements:")
    print(f"  Double DQN: {'Enabled' if use_double_dqn else 'Disabled'}")
    print(f"  Dueling Architecture: {'Enabled' if use_dueling else 'Disabled'}")
    print(f"  Adaptive Epsilon Decay: {epsilon_decay} episodes")
    print("  State Normalization: Enabled")
    
    # Show Q-value statistics if available
    if agent.q_value_history:
        q_values = np.array(agent.q_value_history)
        print(
            f"  Q-value statistics: min={q_values.min():.2f}, "
            f"max={q_values.max():.2f}, mean={q_values.mean():.2f}"
        )
    
    print("\nAdditional Insights:")
    print("- Consider your competitive position based on the competitor analysis")
    print("- Monitor inventory levels to avoid stock-outs")
    print("- Adjust your price sensitivity and promotion boost parameters as you gather more data")
    
    # Print the output directory where all files were saved
    print(f"\nAll plots and analyses saved to: {output_dir}/")
    
    return agent, env, episode_info

# Register the environment with gym
register(
    id='MSME-v0',
    entry_point='gym_working:MSMEEnvironment',
    max_episode_steps=30,  # Default time horizon
)

def create_gym_env(config=None, time_horizon=30):
    import gym
    if config is None:
        config = create_fixed_default_config()
    
    # Try to create the environment directly first
    try:
        # Create the environment
        env = gym.make('MSME-v0', config=config, time_horizon=time_horizon)
    except Exception as e:
        # If that fails, try re-registering the environment
        try:
            # Different Gym versions use different ways to unregister
            if hasattr(gym, 'envs') and hasattr(gym.envs, 'registration'):
                if hasattr(gym.envs.registration, 'registry'):
                    if hasattr(gym.envs.registration.registry, 'env_specs'):
                        # Gym earlier versions
                        gym.envs.registration.registry.env_specs.pop('MSME-v0', None)
                    else:
                        # Some intermediate versions
                        gym.envs.registration.registry.pop('MSME-v0', None)
                elif hasattr(gym.envs, 'registry'):
                    # Current versions
                    gym.envs.registry.pop('MSME-v0', None)
        except Exception:
            # Ignore errors in unregistering
            pass
        
        # Register again
        register(
            id='MSME-v0',
            entry_point='gym_working:MSMEEnvironment',
            max_episode_steps=time_horizon,
        )
        
        # Try to create the environment again
        env = gym.make('MSME-v0', config=config, time_horizon=time_horizon)
    
    return env

def calibrate_reward_shaping(config=None, time_horizon=30, num_episodes=5, num_optimization_trials=50):
    """
    Calibrate reward shaping parameters using Bayesian optimization.
    
    Args:
        config: Configuration for the environment
        time_horizon: Number of time steps per episode
        num_episodes: Number of episodes to run for each parameter setting evaluation
        num_optimization_trials: Number of trials for Bayesian optimization
        
    Returns:
        Tuple of (alpha, beta, gamma, delta) with the best parameter values
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
    except ImportError:
        print("scikit-optimize not installed. Please run: pip install scikit-optimize")
        raise ImportError("scikit-optimize is required for Bayesian optimization")
    
    if config is None:
        config = create_fixed_default_config()
    
    print("Starting Bayesian optimization for reward shaping parameters...")
    
    # Step 1: Run a few episodes with unmodified reward to measure average per-step profit
    env = MSMEEnvironment(config, time_horizon, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0)
    
    episode_profits = []
    
    for _ in range(num_episodes):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        done = False
        episode_profit = 0
        step_count = 0
        
        while not done:
            # Take random actions to explore the environment
            action = env.action_space.sample()
            next_state_tuple, _, terminated, truncated, info = env.step(action)
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            done = terminated or truncated
            episode_profit += info['profit']  # Use base profit, not shaped reward
            step_count += 1
        
        avg_step_profit = episode_profit / step_count if step_count > 0 else 0
        episode_profits.append(avg_step_profit)
    
    avg_step_profit = sum(episode_profits) / len(episode_profits) if episode_profits else 0
    print(f"Baseline average per-step profit: ${avg_step_profit:.2f}")
    
    # Step 2: Define search space as small fractions of average profit
    # Upper bound is 20% of average profit
    upper_bound = 0.2 * avg_step_profit if avg_step_profit > 0 else 0.2
    
    # Define search space (now including delta)
    space = [
        Real(0.0, upper_bound, name='alpha'),  # Holding cost weight
        Real(0.0, upper_bound, name='beta'),   # Stockout penalty weight
        Real(0.0, upper_bound, name='gamma'),  # Price stability weight
        Real(0.0, upper_bound, name='delta')   # Excessive discount penalty weight
    ]
    
    # Step 3: Define objective function
    @use_named_args(space)
    def objective(alpha, beta, gamma, delta):
        # Create environment with current parameters
        env = MSMEEnvironment(config, time_horizon, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        
        # Track metrics across episodes
        all_profits = []
        all_stockout_rates = []
        all_price_changes = []
        all_discount_rates = []  # New: Track excessive discounting
        
        # Run multiple episodes for robust evaluation
        for _ in range(num_episodes):
            state_tuple = env.reset()
            state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
            
            done = False
            episode_profit = 0
            stockout_count = 0
            demand_total = 0
            price_changes = []
            discount_count = 0  # Count periods where price < unit_cost
            total_periods = 0
            prev_price = None
            
            while not done:
                # Take random actions to explore
                action = env.action_space.sample()
                next_state_tuple, _, terminated, truncated, info = env.step(action)
                next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                
                done = terminated or truncated
                
                # Track metrics
                episode_profit += info['profit']  # Use base profit for evaluation
                stockout_count += info['unfulfilled']
                demand_total += info['demand']
                
                # Track price changes
                if prev_price is not None:
                    price_changes.append(abs(info['price'] - prev_price))
                prev_price = info['price']
                
                # Track excessive discounting
                if info['price'] < config.unit_cost:
                    discount_count += 1
                total_periods += 1
            
            # Calculate episode-level metrics
            all_profits.append(episode_profit)
            
            
            # Calculate stockout rate for the episode
            stockout_rate = stockout_count / demand_total if demand_total > 0 else 0
            all_stockout_rates.append(stockout_rate)
            
            # Calculate average price change for the episode
            avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
            all_price_changes.append(avg_price_change)
        
        # Calculate averages across all episodes
        avg_profit = sum(all_profits) / len(all_profits) if all_profits else 0
        avg_stockout_rate = sum(all_stockout_rates) / len(all_stockout_rates) if all_stockout_rates else 0
        avg_price_change = sum(all_price_changes) / len(all_price_changes) if all_price_changes else 0
        
        # Calculate composite score (higher is better)
        # 70% profit, 15% stockout minimization, 15% price stability
        score = (0.7 * avg_profit) - (0.15 * avg_stockout_rate * 1000) - (0.15 * avg_price_change * 100)
        
        # For minimization, return negative score
        return -score
    
    # Step 4: Run Bayesian optimization
    print(f"Running Bayesian optimization with {num_optimization_trials} trials...")
    
    # Ensure n_initial_points is not greater than num_optimization_trials
    n_initial_points = min(5, num_optimization_trials)
    
    result = gp_minimize(
        objective,
        space,
        n_calls=num_optimization_trials,
        n_initial_points=n_initial_points,
        verbose=True,
        random_state=42
    )
    
    # Extract best parameters
    best_alpha, best_beta, best_gamma, best_delta = result.x
    best_score = -result.fun  # Convert back to positive score
    
    print("\nOptimization complete.")
    print(f"Best parameters: alpha={best_alpha:.4f}, beta={best_beta:.4f}, gamma={best_gamma:.4f}, delta={best_delta:.4f}")
    print(f"Best score: {best_score:.2f}")
    
    return best_alpha, best_beta, best_gamma, best_delta

def grid_search_calibration(config=None, time_horizon=30, num_episodes=5, num_search_episodes=20):
    """
    Fallback grid search for reward shaping parameters if scikit-optimize is not available.
    """
    if config is None:
        config = create_fixed_default_config()
    
    # Run a few episodes with unmodified reward to measure average per-step profit
    env = MSMEEnvironment(config, time_horizon, alpha=1.0, beta=1.0, gamma=0.0)
    
    episode_rewards = []
    per_step_profits = []
    
    for _ in range(num_episodes):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        done = False
        episode_reward = 0
        step_count = 0
        total_profit = 0
        
        while not done:
            # Take random actions to explore the environment
            action = env.action_space.sample()
            next_state_tuple, reward, terminated, truncated, info = env.step(action)
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            done = terminated or truncated
            episode_reward += reward
            total_profit += info['profit']
            step_count += 1
        
        episode_rewards.append(episode_reward)
        per_step_profit = total_profit / step_count if step_count > 0 else 0
        per_step_profits.append(per_step_profit)
    
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    avg_step_profit = sum(per_step_profits) / len(per_step_profits) if per_step_profits else 0
    
    print(f"Average per-step profit: ${avg_step_profit:.2f}")
    
    # Initialize shaping parameters as a percentage of average per-step profit
    init_alpha = 0.075 * avg_step_profit  # 7.5% of avg profit
    init_beta = 0.03 * avg_step_profit    # 3% of avg profit
    init_gamma = 0.03 * avg_step_profit   # 3% of avg profit
    
    print(f"Initial reward shaping parameters:")
    print(f"α (holding cost weight): ${init_alpha:.2f}")
    print(f"β (stockout penalty weight): ${init_beta:.2f}")
    print(f"γ (price stability weight): ${init_gamma:.2f}")
    
    # Grid search around these values
    alpha_range = [init_alpha * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
    beta_range = [init_beta * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
    gamma_range = [init_gamma * factor for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
    
    # Store results
    results = []
    
    # Count total iterations for progress tracking
    total_iterations = len(alpha_range) * len(beta_range) * len(gamma_range)
    current_iteration = 0
    
    print(f"Running grid search with {total_iterations} parameter combinations...")
    
    # Try different combinations of parameters
    for alpha in alpha_range:
        for beta in beta_range:
            for gamma in gamma_range:
                current_iteration += 1
                if current_iteration % 5 == 0:
                    print(f"Progress: {current_iteration}/{total_iterations} combinations tested")
                
                # Create environment with current shaping parameters
                env = MSMEEnvironment(config, time_horizon, alpha=alpha, beta=beta, gamma=gamma)
                
                # Track metrics across episodes
                total_profit = 0
                total_stockout_rate = 0
                total_price_change = 0
                
                # Run multiple episodes for robust evaluation
                for _ in range(num_search_episodes):
                    state_tuple = env.reset()
                    state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
                    
                    done = False
                    episode_profit = 0
                    stockout_count = 0
                    demand_total = 0
                    price_changes = []
                    prev_price = None
                    step_count = 0
                    
                    while not done:
                        # Take random actions to explore the environment
                        action = env.action_space.sample()
                        next_state_tuple, _, terminated, truncated, info = env.step(action)
                        next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
                        
                        done = terminated or truncated
                        
                        # Track metrics
                        episode_profit += info['profit']  # Use base profit, not shaped reward
                        stockout_count += info['unfulfilled']
                        demand_total += info['demand']
                        
                        # Track price changes
                        if prev_price is not None:
                            price_changes.append(abs(info['price'] - prev_price))
                        prev_price = info['price']
                        
                        step_count += 1
                    
                    # Calculate episode metrics
                    total_profit += episode_profit
                    stockout_rate = stockout_count / demand_total if demand_total > 0 else 0
                    total_stockout_rate += stockout_rate
                    avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
                    total_price_change += avg_price_change
                
                # Calculate average metrics across episodes
                avg_profit = total_profit / num_search_episodes
                avg_stockout_rate = total_stockout_rate / num_search_episodes
                avg_price_change = total_price_change / num_search_episodes
                
                # Calculate a composite score (higher is better)
                # Prioritize profit (70%) but also reward low stockouts (15%) and price stability (15%)
                score = (0.7 * avg_profit) - (0.15 * avg_stockout_rate * 1000) - (0.15 * avg_price_change * 100)
                
                # Store result
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'avg_profit': avg_profit,
                    'avg_stockout_rate': avg_stockout_rate,
                    'avg_price_change': avg_price_change,
                    'score': score
                })
    
    # Sort results by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Get best parameters
    best_params = (results[0]['alpha'], results[0]['beta'], results[0]['gamma'])
    
    # Print summary
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5]):
        print(f"Rank {i+1}: α={result['alpha']:.4f}, β={result['beta']:.4f}, γ={result['gamma']:.4f}")
        print(f"  Avg Profit: ${result['avg_profit']:.2f}, Stockout Rate: {result['avg_stockout_rate']*100:.1f}%, "
              f"Avg Price Change: ${result['avg_price_change']:.2f}")
    
    print(f"\nSelected parameters: α={best_params[0]:.4f}, β={best_params[1]:.4f}, γ={best_params[2]:.4f}")
    
    return best_params

def train_with_gym_interface(config=None, num_episodes=300, print_every=10, calibrate_rewards=True, 
                            alpha=0.0, beta=0.0, gamma=0.0, delta=0.0,
                            use_double_dqn=True, use_dueling=True):
    """Train a DQN agent using the standard gym interface, optionally with reward shaping"""
    # Create or use the provided config
    if config is None:
        config = create_fixed_default_config()
    
    print("Setting up training environment...")
    
    # Check if manual parameters are provided
    use_manual_params = alpha > 0 or beta > 0 or gamma > 0 or delta > 0
    
    if calibrate_rewards and not use_manual_params:
        print("Calibrating reward shaping parameters using Bayesian optimization...")
        # Directly call the calibration function without try-except
        alpha, beta, gamma, delta = calibrate_reward_shaping(
            config=config, 
            time_horizon=30,
            num_episodes=5,
            num_optimization_trials=50
        )
        print(f"Using calibrated parameters: α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, δ={delta:.4f}")
    elif not calibrate_rewards and not use_manual_params:
        # Use default values if no calibration and no manual params
        alpha, beta, gamma, delta = 1.0, 1.0, 0.2, 0.2
        print(f"Using default reward shaping parameters: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    else:
        print(f"Using manual reward shaping parameters: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    
    # Create environment with calibrated parameters
    env = MSMEEnvironment(config, time_horizon=30, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    
    # Setup device and other parameters
    batch_size = 64
    gamma = 0.95
    epsilon_start = 0.9
    epsilon_end = 0.05
    # Adaptive epsilon decay based on episode count
    epsilon_decay = num_episodes // 3
    target_update = 10
    memory_size = 10000
    learning_rate = 0.001
    hidden_size = 128
    reward_scale = 100.0  # Scale factor for rewards
    
    # Get state and action dimensions - RESET THE ENVIRONMENT FIRST to get correct dimensions
    obs_tuple = env.reset()
    # Handle both formats: just state or (state, info) tuple
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    
    state_size = len(obs)
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create networks
    policy_net = DQNPricingNetwork(
        state_size, 
        action_size, 
        hidden_size, 
        dueling=use_dueling
    ).to(device)
    
    target_net = DQNPricingNetwork(
        state_size, 
        action_size, 
        hidden_size, 
        dueling=use_dueling
    ).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_size)
    
    # Training stats
    steps_done = 0
    episodes_done = 0
    all_rewards = []
    all_losses = []
    best_episode_reward = -float('inf')
    q_value_history = []
    
    # Define the action selection function
    def select_action(state, evaluate=False):
        if evaluate:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                
                # Track Q-values for diagnostics
                if len(q_value_history) < 1000:
                    q_value_history.append(q_values.max().item())
                
                return q_values.max(1)[1].item()
        
        nonlocal steps_done
        sample = random.random()
        # Use episode-based decay instead of step-based
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
                      math.exp(-1. * episodes_done / epsilon_decay)
        steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                
                # Track Q-values for diagnostics
                if len(q_value_history) < 1000:
                    q_value_history.append(q_values.max().item())
                
                return q_values.max(1)[1].item()
        else:
            return random.randrange(action_size)
    
    # Training loop
    for episode in range(num_episodes):
        episodes_done = episode  # Update for epsilon calculation
        
        # Reset environment
        obs_tuple = env.reset()
        # Handle both formats: just state or (state, info) tuple
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        # Initialize episode variables
        episode_reward = 0
        episode_losses = []
        done = False
        
        # Run episode
        while not done:
            # Select and perform an action
            action = select_action(obs)
            next_obs_tuple, reward, terminated, truncated, info = env.step(action)
            # Handle both formats: just state or (state, info) tuple
            next_obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple
            
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition in memory
            memory.push(obs, action, next_obs if not done else None, reward, float(done))
            
            # Move to next state
            obs = next_obs
            
            # Perform optimization if enough samples
            if len(memory) >= batch_size:
                # Sample batch from memory
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                
                # Create mask for non-final states
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                     batch.next_state)), 
                                           device=device, dtype=torch.bool)
                
                non_final_next_states = torch.cat([torch.FloatTensor(s).unsqueeze(0) 
                                                 for s in batch.next_state 
                                                 if s is not None]).to(device)
                
                state_batch = torch.FloatTensor(batch.state).to(device)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor(batch.reward).to(device)
                done_batch = torch.FloatTensor(batch.done).to(device)
                
                # Scale rewards for stability
                scaled_reward_batch = reward_batch / reward_scale
                
                # Compute Q(s_t, a)
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                
                # Compute V(s_{t+1}) - using Double DQN if enabled
                next_state_values = torch.zeros(batch_size, device=device)
                if non_final_mask.sum() > 0:
                    if use_double_dqn:
                        # Double DQN: Use policy net to select actions
                        next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                        # Use target net to evaluate values
                        next_state_values[non_final_mask] = target_net(
                            non_final_next_states
                        ).gather(1, next_actions).detach().squeeze()
                    else:
                        # Standard DQN: Use target net for both selection and evaluation
                        next_state_values[non_final_mask] = target_net(
                            non_final_next_states
                        ).max(1)[0].detach()
                
                # Compute expected Q values
                expected_state_action_values = (
                    scaled_reward_batch + gamma * next_state_values * (1 - done_batch)
                )
                
                # Compute Huber loss
                loss = F.smooth_l1_loss(
                    state_action_values, 
                    expected_state_action_values.unsqueeze(1)
                )
                episode_losses.append(loss.item())
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Store episode rewards and losses
        all_rewards.append(episode_reward)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
        all_losses.append(avg_loss)
        
        # Track best episode
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
        
        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(all_rewards[-print_every:]) if len(all_rewards) >= print_every else np.mean(all_rewards)
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * episodes_done / epsilon_decay)
            print(f"Episode {episode}/{num_episodes} | Avg Reward: ${avg_reward:.2f} | Epsilon: {epsilon:.2f}")
    
    print(f"Training complete! Best episode reward: ${best_episode_reward:.2f}")
    
    # Print Q-value statistics if available
    if q_value_history:
        q_values = np.array(q_value_history)
        print(f"Q-value statistics: min={q_values.min():.2f}, max={q_values.max():.2f}, mean={q_values.mean():.2f}")
    
    # Create agent using the SimpleAgent class defined at the module level
    agent = SimpleAgent(policy_net)
    
    return agent, env, all_rewards, all_losses

# Define SimpleAgent outside of any function
class SimpleAgent:
    """A simple agent wrapper for a trained policy network"""
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.device = next(policy_net.parameters()).device
        
    def select_action(self, state, evaluate=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_tensor).max(1)[1].item()

def train_with_stable_baselines(config=None, total_timesteps=100000):
    """Train using Stable-Baselines3 DQN"""
    try:
        # Import Stable-Baselines3
        from stable_baselines3 import DQN
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        print("Stable-Baselines3 not installed. Please run: pip install stable-baselines3")
        return None

    # Create the environment
    env = create_gym_env(config)
    
    # Wrap with standard SB3 approach
    try:
        # Try wrapping directly
        env = make_vec_env(lambda: env, n_envs=1)
        
        # Create the DQN agent
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            target_update_interval=10,
            verbose=1
        )
        
        # Train the agent
        print("Training model using Stable-Baselines3 DQN...")
        model.learn(total_timesteps=total_timesteps)
        
        # Save the model
        model.save("msme_dqn_model")
        print("Model saved as msme_dqn_model")
        
        # Evaluate the trained agent
        mean_reward = evaluate_trained_model(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f}")
        
        return model, env
    
    except Exception as e:
        print(f"Error with Stable-Baselines3: {e}")
        print("This may be due to gym/gymnasium compatibility issues.")
        import traceback
        traceback.print_exc()
        return None, env

def evaluate_trained_model(model, env, n_eval_episodes=10):
    """Custom evaluation function for SB3 models"""
    episode_rewards = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_rewards.append(total_reward)
    
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    return mean_reward

def evaluate_and_visualize_gym_env(model=None, config=None, num_episodes=5, use_random=False):
    """Run a few episodes and visualize results"""
    # Create the environment
    env = create_gym_env(config)
    
    # Try to make it record videos if possible
    try:
        from gym.wrappers import Monitor
        try:
            env = Monitor(env, "./videos/", force=True, video_callable=lambda episode_id: True)
        except Exception as e:
            print(f"Warning: Could not wrap with Monitor: {e}")
    except ImportError:
        pass
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    episode_rewards = []
    all_info = []
    all_time_steps = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        # Reset environment
        obs_tuple = env.reset()
        # Handle both formats: just state or (state, info) tuple
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        step = 0
        total_reward = 0
        done = False
        episode_info = []
        time_steps = []
        
        # Run episode
        while not done:
            if use_random:
                # Use random actions
                action = env.action_space.sample()
            else:
                # Use trained agent if provided
                if model is not None:
                    # Check if it's a Stable Baselines model with predict method
                    if hasattr(model, 'predict'):
                        action, _ = model.predict(obs, deterministic=True)
                    # Check if it's our custom agent with select_action method
                    elif hasattr(model, 'select_action'):
                        action = model.select_action(obs, evaluate=True)
                    # Otherwise use the policy_net directly
                    else:
                        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        with torch.no_grad():
                            q_values = model.policy_net(state_tensor)
                            action = q_values.max(1)[1].item()
                else:
                    # Use heuristic (pricing similar to competitor)
                    if step == 0:
                        action = len(env.env.config.price_tiers) // 2  # Middle price tier
                    else:
                        # If we have previous step info, adapt to competitor
                        comp_price = episode_info[-1]['comp_price']
                        price_tiers = env.env.config.price_tiers
                        # Find closest price tier to competitor price
                        action = np.abs(price_tiers - comp_price).argmin()
            
            # Take step
            next_obs_tuple, reward, terminated, truncated, info = env.step(action)
            # Handle both formats: just state or (state, info) tuple
            next_obs = next_obs_tuple[0] if isinstance(next_obs_tuple, tuple) else next_obs_tuple
            
            # Record info
            episode_info.append(info)
            time_steps.append(step)
            
            # Move to next step
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
            step += 1
        
        # Record episode results
        episode_rewards.append(total_reward)
        all_info.append(episode_info)
        all_time_steps.append(time_steps)
        
        print(f"Episode {episode+1}: Total Reward = ${total_reward:.2f}, Steps = {step}")
    
    # Calculate summary statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: ${avg_reward:.2f}")
    
    # Plot summary
    plt.figure(figsize=(12, 10))
    
    # Plot rewards across episodes
    plt.subplot(3, 1, 1)
    plt.bar(range(1, num_episodes+1), episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot prices for the first episode
    plt.subplot(3, 1, 2)
    prices = [info['price'] for info in all_info[0]]
    plt.plot(all_time_steps[0], prices, 'b-', linewidth=2)
    plt.title('Price Strategy (First Episode)')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    # Plot inventory for the first episode
    plt.subplot(3, 1, 3)
    inventory = [info['inventory'] for info in all_info[0]]
    plt.plot(all_time_steps[0], inventory, 'g-', linewidth=2)
    plt.title('Inventory Level (First Episode)')
    plt.xlabel('Time Step')
    plt.ylabel('Inventory')
    
    plt.tight_layout()
    plt.savefig('evaluation_summary.png')
    plt.show()
    
    return env, episode_rewards, all_info, all_time_steps

# Main demonstration script
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='MSME Pricing Environment Demo')
    parser.add_argument('--mode', type=str, default='custom', 
                      choices=['custom', 'gym', 'sb3', 'evaluate_random', 'evaluate_model', 'test_env'],
                      help='Mode to run: custom (original), gym (standard interface), sb3 (stable-baselines3), evaluate')
    parser.add_argument('--episodes', type=int, default=300, help='Number of episodes for training')
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of timesteps for SB3 training')
    parser.add_argument('--model_path', type=str, default='msme_dqn_model', help='Path to saved model for evaluation')
    parser.add_argument('--no_calibrate', action='store_true', help='Disable reward shaping calibration')
    parser.add_argument('--alpha', type=float, default=0.0, help='Holding cost penalty weight (0 to use calibrated value)')
    parser.add_argument('--beta', type=float, default=0.0, help='Stockout penalty weight (0 to use calibrated value)')
    parser.add_argument('--gamma', type=float, default=0.0, help='Price stability penalty weight (0 to use calibrated value)')
    parser.add_argument('--delta', type=float, default=0.0, help='Excessive discount penalty weight (0 to use calibrated value)')
    parser.add_argument('--no_double_dqn', action='store_true', help='Disable Double DQN (use standard DQN)')
    parser.add_argument('--no_dueling', action='store_true', help='Disable Dueling network architecture')
    
    args = parser.parse_args()
    
    # Create or select configuration
    if args.mode == 'test_env':
        # Just test the basic environment functionality
        try:
            print("Testing environment functionality...")
            
            # Create a simple environment
            config = create_fixed_default_config()
            gym_env = create_gym_env(config)
            
            # Take a single step and check if it works
            obs, info = gym_env.reset()
            action = gym_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            print(f"Gym environment functional: step returned reward {reward}")
            
        except Exception as e:
            print(f"Error during environment test: {e}")
        
        sys.exit(0)
    
    # Otherwise initialize a config
    config = create_fixed_default_config()
    
    if args.mode == 'custom':
        # Run the original MSME model
        print("Running original MSME model...")
        agent, env, episode_info = run_msme_dqn_model()
    
    elif args.mode == 'gym':
        # Train using standard gym interface
        print("Training with gym interface...")
        
        # Check for manual parameters
        use_manual_params = args.alpha > 0 or args.beta > 0 or args.gamma > 0 or args.delta > 0
        
        if use_manual_params:
            print(f"Using manual reward shaping parameters: α={args.alpha}, β={args.beta}, γ={args.gamma}, δ={args.delta}")
            # Don't create the environment here, let train_with_gym_interface create it
            agent, env, rewards, losses = train_with_gym_interface(
                config=config, 
                num_episodes=args.episodes,
                calibrate_rewards=not args.no_calibrate,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                use_double_dqn=not args.no_double_dqn,
                use_dueling=not args.no_dueling
            )
        else:
            # Use calibration
            agent, env, rewards, losses = train_with_gym_interface(
                config=config, 
                num_episodes=args.episodes,
                calibrate_rewards=not args.no_calibrate,
                use_double_dqn=not args.no_double_dqn,
                use_dueling=not args.no_dueling
            )
        
        # Plot training progress
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        rewards_smoothed = pd.Series(rewards).rolling(window=10).mean()
        plt.plot(rewards, alpha=0.3)
        plt.plot(rewards_smoothed)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(2, 2, 2)
        if losses:
            losses_smoothed = pd.Series(losses).rolling(window=10).mean()
            plt.plot(losses, alpha=0.3)
            plt.plot(losses_smoothed)
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
        
        # Evaluate the trained agent
        print("\nEvaluating trained agent...")
        evaluate_and_visualize_gym_env(model=agent, config=config, num_episodes=3)
    
    elif args.mode == 'sb3':
        # Train with Stable-Baselines3
        try:
            print("Training with Stable-Baselines3...")
            model, env = train_with_stable_baselines(
                config=config, 
                total_timesteps=args.timesteps
            )
            
            if model is not None:
                # Save the model
                model.save("sb3_msme_model")
                
                # Evaluate the trained model
                print("\nEvaluating trained SB3 model...")
                evaluate_and_visualize_gym_env(model=model, config=config, num_episodes=3)
        except Exception as e:
            print(f"Error during SB3 training: {e}")
    
    elif args.mode == 'evaluate_random':
        # Evaluate with random actions
        print("Evaluating environment with random actions...")
        evaluate_and_visualize_gym_env(config=config, num_episodes=5, use_random=True)
    
    elif args.mode == 'evaluate_model':
        # Load and evaluate a saved model
        try:
            from stable_baselines3 import DQN
            model = DQN.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
            
            evaluate_and_visualize_gym_env(model=model, config=config, num_episodes=3)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to random evaluation")
            evaluate_and_visualize_gym_env(config=config, num_episodes=3, use_random=True)
    
    print("\nDemo complete")

# Example of how to train and evaluate the agent
"""
# Example usage:
# 1. Train the agent first
agent, env, episode_info = run_msme_dqn_model()

# 2. Then evaluate it with the proper visualizations
evaluate_and_visualize_gym_env(
    model=agent,              # Pass your trained agent here
    config=env.config,        # Use the same config
    num_episodes=1,           # Just one episode to see the visualization 
    use_random=False          # Don't use random actions
)
"""




