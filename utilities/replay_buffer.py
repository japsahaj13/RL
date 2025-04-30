"""
Replay Buffer implementation for experience replay in Deep RL algorithms.
"""

import random
from collections import namedtuple, deque
import torch
import numpy as np
from typing import Tuple, List, Any, Optional

# Define transition tuple for storing experiences
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """
    A cyclic buffer of bounded size that holds the transitions observed recently.
    
    This implementation uses a deque for efficient append and pop operations.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer with a maximum capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args) -> None:
        """
        Save a transition to the buffer.
        
        Args:
            *args: Arguments to construct a Transition tuple
        """
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            A list of sampled Transition tuples
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        return len(self.memory)


class PrioritizedReplayMemory:
    """
    A prioritized replay buffer that samples transitions based on their priorities.
    
    This implementation uses a sum tree for efficient sampling based on priorities.
    """
    
    def __init__(
            self, 
            capacity: int = 10000, 
            alpha: float = 0.6, 
            beta_start: float = 0.4, 
            beta_frames: int = 100000
    ):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform sampling, higher = more prioritization)
            beta_start: Initial beta value for importance sampling correction
            beta_frames: Number of frames over which to anneal beta from beta_start to 1.0
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
    def push(self, *args) -> None:
        """
        Save a transition to the buffer with maximum priority.
        
        Args:
            *args: Arguments to construct a Transition tuple
        """
        max_priority = max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], torch.Tensor, List[int]]:
        """
        Sample a batch of transitions based on their priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (transitions, importance_weights, indices)
        """
        if len(self.memory) == 0:
            return [], torch.tensor([]), []
        
        # Calculate current beta for importance sampling
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Get the sampled transitions
        transitions = [self.memory[idx] for idx in indices]
        
        return transitions, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        return len(self.memory)


def get_batch_tensor(
        transitions: List[Transition], 
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of Transition tuples to tensors for neural network processing.
    
    Args:
        transitions: List of Transition tuples
        device: Device to move tensors to
        
    Returns:
        Tuple of (state_batch, action_batch, next_state_batch, reward_batch, done_batch)
    """
    batch = Transition(*zip(*transitions))
    
    # Convert to tensors and move to device
    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long).view(-1, 1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float).view(-1, 1).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float).view(-1, 1).to(device)
    
    # Handle terminal states (next_state can be None for terminal states)
    non_terminal_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device, 
        dtype=torch.bool
    )
    
    # Only include non-terminal next states
    non_terminal_next_states = torch.cat([
        s.unsqueeze(0) for s in batch.next_state if s is not None
    ]).to(device)
    
    next_state_batch = torch.zeros_like(state_batch)
    next_state_batch[non_terminal_mask] = non_terminal_next_states
    
    return state_batch, action_batch, next_state_batch, reward_batch, done_batch 