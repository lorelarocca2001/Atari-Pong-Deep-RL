
import torch

from src.training.dqn_agent import DQNAgent


def test_dqn_agent():
    """
    Verify target network initialization, freezing, and synchronization.

    Checks that target parameters remain unchanged when the online
    network is modified and are updated only after synchronization.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(num_actions=3, device=device)

    online_parameters = list(agent.online_network.parameters())
    target_parameters = list(agent.target_network.parameters())

    assert all(
        torch.equal(online, target)
        for online, target in zip(online_parameters, target_parameters)
    )

    assert all(not parameter.requires_grad for parameter in target_parameters)

    print("Initial networks identical: True")
    print("Target network frozen: True")

    original_target = [parameter.clone() for parameter in target_parameters]

    with torch.no_grad():
        for parameter in online_parameters:
            parameter.add_(0.1)

    assert all(
        torch.equal(original, current)
        for original, current in zip(original_target, target_parameters)
    )

    assert any(
        not torch.equal(online, target)
        for online, target in zip(online_parameters, target_parameters)
    )

    print("Target unchanged after online update: True")

    agent.update_target_network()

    assert all(
        torch.equal(online, target)
        for online, target in zip(online_parameters, target_parameters)
    )

    print("Networks identical after synchronization: True")
    print("Device:", device)
    print("DQN agent test passed!")


if __name__ == "__main__":
    test_dqn_agent()
