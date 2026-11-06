
import numpy as np
import torch
import torch.nn as nn

from src.training.dqn_agent import DQNAgent
from src.datasets.replay_buffer import ReplayBuffer


def test_dqn_training():
    """
    Verify DQN optimization using a sampled replay buffer minibatch.

    Checks finite MSE loss, online network parameter updates,
    and target network stability.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(
        num_actions=3,
        device=device,
        learning_rate=1e-4,
        gamma=0.99
    )

    assert isinstance(agent.loss_function, nn.MSELoss)

    buffer = ReplayBuffer(capacity=100)

    for i in range(64):
        state = np.random.randint(
            0, 256, (4, 84, 84), dtype=np.uint8
        )
        next_state = np.random.randint(
            0, 256, (4, 84, 84), dtype=np.uint8
        )

        buffer.push(
            state=state,
            action=i % 3,
            reward=float(i % 3 - 1),
            next_state=next_state,
            done=i % 10 == 0
        )

    batch = buffer.sample(batch_size=32, device=device)

    online_before = [
        parameter.detach().clone()
        for parameter in agent.online_network.parameters()
    ]

    target_before = [
        parameter.detach().clone()
        for parameter in agent.target_network.parameters()
    ]

    loss = agent.train_step(batch)

    online_after = list(agent.online_network.parameters())
    target_after = list(agent.target_network.parameters())

    assert np.isfinite(loss)
    assert loss >= 0

    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, online_after)
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, target_after)
    )

    print("Device:", device)
    print("Loss function: MSE")
    print("TD Loss:", loss)
    print("Online network updated: True")
    print("Target network unchanged: True")
    print("DQN training test passed!")


if __name__ == "__main__":
    test_dqn_training()
