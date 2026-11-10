


import torch
import torch.nn as nn


class DQN(nn.Module):
    """Convolutional Deep Q-Network for Atari Pong."""

    def __init__(
        self,
        num_actions: int = 3,
        activation: str = "leaky_relu"
    ):
        """
        Initialize the convolutional Q-network.

        Args:
            num_actions: Number of discrete actions available to the agent.
            activation: Activation function for the last convolutional layer.
        """
        super().__init__()

        if activation == "relu":
            final_activation = nn.ReLU()
        elif activation == "leaky_relu":
            final_activation = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            final_activation
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each action.

        Args:
            x: Batch of stacked frames with shape (B, 4, 84, 84).

        Returns:
            Q-values with shape (B, num_actions).
        """
        x = x.float() / 255.0
        x = self.features(x)
        return self.classifier(x)


