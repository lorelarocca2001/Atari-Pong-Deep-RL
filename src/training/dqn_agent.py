
import torch

from src.models.dqn import DQN


class DQNAgent:
    """Manage the online and target networks used by the DQN algorithm."""

    def __init__(self, num_actions: int, device: torch.device):
        """
        Initialize two identical Q-networks.

        Args:
            num_actions: Number of discrete actions.
            device: Device used for neural network computations.
        """
        self.device = device

        self.online_network = DQN(num_actions).to(device)
        self.target_network = DQN(num_actions).to(device)

        self.update_target_network()
        self.target_network.eval()

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

    def update_target_network(self) -> None:
        """Copy online network parameters to the target network."""
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
