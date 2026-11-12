
import random

import numpy as np
import torch
import torch.nn as nn

from src.models.dqn import DQN


class DQNAgent:
    """Manage DQN networks, optimization, and epsilon-greedy action selection."""

    def __init__(
        self,
        num_actions: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        activation: str = "leaky_relu",
        double_dqn: bool = False
    ):
        """
        Initialize the online and target networks.

        Args:
            num_actions: Number of discrete actions.
            device: Device used for neural network computations.
            learning_rate: Learning rate for the Adam optimizer.
            gamma: Discount factor for future rewards.
            activation: Activation function for the last convolutional layer.
            double_dqn: Whether to use Double DQN targets.
        """
        self.device = device
        self.gamma = gamma
        self.num_actions = num_actions
        self.activation = activation
        self.double_dqn = double_dqn

        self.online_network = DQN(
            num_actions=num_actions,
            activation=activation
        ).to(device)

        self.target_network = DQN(
            num_actions=num_actions,
            activation=activation
        ).to(device)

        self.update_target_network()
        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

    def update_target_network(self) -> None:
        """Copy online network parameters to the target network."""
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def train_step(self, batch: tuple) -> float:
        """
        Perform one optimization step using DQN or Double DQN targets.

        Args:
            batch: Tuple of states, actions, rewards, next_states, and dones.

        Returns:
            Scalar Temporal Difference loss.
        """
        states, actions, rewards, next_states, dones = batch

        self.online_network.train()

        q_values = self.online_network(states)
        current_q_values = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online_network(
                    next_states
                ).argmax(dim=1, keepdim=True)

                next_q_values = self.target_network(
                    next_states
                ).gather(
                    1, next_actions
                ).squeeze(1)
            else:
                next_q_values = self.target_network(
                    next_states
                ).max(dim=1).values

            targets = (
                rewards
                + self.gamma * (1.0 - dones) * next_q_values
            )

        loss = self.loss_function(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select an action using the epsilon-greedy policy.

        Args:
            state: Current stacked observation with shape (4, 84, 84).
            epsilon: Probability of selecting a random action.

        Returns:
            Selected discrete action index.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, device=self.device
            ).unsqueeze(0)

            q_values = self.online_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return int(action)


