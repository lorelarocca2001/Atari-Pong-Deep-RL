
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    """Represent a single environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Store transitions and sample random minibatches for DQN training."""

    def __init__(self, capacity: int):
        """
        Initialize a fixed-capacity experience replay buffer.

        Args:
            capacity: Maximum number of transitions stored.
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.append(
            Transition(state.copy(), action, reward, next_state.copy(), done)
        )

    def sample(self, batch_size: int, device: torch.device) -> tuple:
        """
        Sample a random minibatch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            device: Device on which to place the tensors.

        Returns:
            Tuple of state, action, reward, next-state, and done tensors.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.as_tensor(
            np.stack([t.state for t in batch]), device=device
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long, device=device
        )
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        )
        next_states = torch.as_tensor(
            np.stack([t.next_state for t in batch]), device=device
        )
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)
