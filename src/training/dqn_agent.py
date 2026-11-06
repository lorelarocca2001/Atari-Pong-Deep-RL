
import torch
import torch.nn as nn

from src.models.dqn import DQN


class DQNAgent:
    """Manage DQN networks and optimize action-value estimates."""

    def __init__(
        self,
        num_actions: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99
    ):
        """
        Initialize the online and target networks.

        Args:
            num_actions: Number of discrete actions.
            device: Device used for neural network computations.
            learning_rate: Learning rate for the Adam optimizer.
            gamma: Discount factor for future rewards.
        """
        self.device = device
        self.gamma = gamma

        self.online_network = DQN(num_actions).to(device)
        self.target_network = DQN(num_actions).to(device)

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
        Perform one DQN optimization step using a sampled minibatch.

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
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1).values

            targets = (
                rewards
                + self.gamma * (1.0 - dones) * max_next_q_values
            )

        loss = self.loss_function(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
