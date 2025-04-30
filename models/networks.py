"""
Neural Network architectures for the MSME Pricing RL project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNPricingNetwork(nn.Module):
    """
    Deep Q-Network for pricing decisions.
    
    This network can be configured as a standard DQN or as a dueling DQN.
    The dueling architecture separates value and advantage streams,
    which can lead to better policy evaluation.
    """
    
    def __init__(
            self, 
            state_size: int, 
            action_size: int, 
            hidden_size: int = 128, 
            dueling: bool = True
    ):
        """
        Initialize the DQN Pricing Network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
            dueling: Whether to use dueling network architecture
        """
        super(DQNPricingNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        if dueling:
            # Dueling architecture
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
            # Standard architecture
            self.output = nn.Linear(hidden_size, action_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.dueling:
            # Dueling architecture combines value and advantage
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Combine value and advantage using the dueling formula
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN output
            return self.output(x)
    
    def save(self, path: str) -> None:
        """
        Save the model weights.
        
        Args:
            path: Path to save the model
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save({
            'state_dict': self.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'dueling': self.dueling
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'DQNPricingNetwork':
        """
        Load a model from a saved file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to (default: current device)
            
        Returns:
            Loaded DQNPricingNetwork
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        state_size = checkpoint['state_size']
        action_size = checkpoint['action_size']
        dueling = checkpoint.get('dueling', True)
        
        model = cls(state_size, action_size, dueling=dueling)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        return model


class A2CPricingNetwork(nn.Module):
    """
    Actor-Critic Network for pricing decisions.
    
    This network has two heads:
    - The actor head outputs action probabilities
    - The critic head outputs state value estimates
    """
    
    def __init__(
            self, 
            state_size: int, 
            action_size: int, 
            hidden_size: int = 128
    ):
        """
        Initialize the A2C Pricing Network.
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            hidden_size: Size of hidden layers
        """
        super(A2CPricingNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        shared_features = self.shared(x)
        
        # Actor: action probabilities
        action_logits = self.actor(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: state value
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
    
    def save(self, path: str) -> None:
        """
        Save the model weights.
        
        Args:
            path: Path to save the model
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save({
            'state_dict': self.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'A2CPricingNetwork':
        """
        Load a model from a saved file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to (default: current device)
            
        Returns:
            Loaded A2CPricingNetwork
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        state_size = checkpoint['state_size']
        action_size = checkpoint['action_size']
        
        model = cls(state_size, action_size)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        return model 