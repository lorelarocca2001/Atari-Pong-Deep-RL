
import numpy as np
import torch

from src.evaluation.evaluate import evaluate_agent
from src.training.dqn_agent import DQNAgent


def test_evaluate():
    """
    Verify DQN evaluation metrics and network parameter stability.

    Checks reward statistics, episode count, and preservation
    of network parameters and training mode.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(num_actions=3, device=device)
    agent.online_network.train()

    parameters_before = [
        parameter.detach().clone()
        for parameter in agent.online_network.parameters()
    ]

    results = evaluate_agent(
        agent=agent,
        num_episodes=2,
        epsilon=0.01
    )

    parameters_after = list(agent.online_network.parameters())

    assert len(results["episode_rewards"]) == 2

    assert np.isclose(
        results["mean_reward"],
        np.mean(results["episode_rewards"])
    )

    assert np.isclose(
        results["std_reward"],
        np.std(results["episode_rewards"])
    )

    assert all(
        -21 <= reward <= 21
        for reward in results["episode_rewards"]
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )

    assert agent.online_network.training

    print("Device:", device)
    print("Episode rewards:", results["episode_rewards"])
    print("Mean reward:", results["mean_reward"])
    print("Reward standard deviation:", results["std_reward"])
    print("Network parameters unchanged: True")
    print("Training mode restored: True")
    print("DQN evaluation test passed!")


if __name__ == "__main__":
    test_evaluate()
