
import numpy as np
import torch

from src.training.dqn_agent import DQNAgent


def test_epsilon_greedy():
    """
    Verify epsilon-greedy exploration and exploitation.

    Checks random action selection, greedy policy behavior,
    and epsilon validation.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(num_actions=3, device=device)
    agent.online_network.eval()

    state = np.random.randint(
        0, 256, (4, 84, 84), dtype=np.uint8
    )

    # Verify pure exploration produces valid and varied actions.
    random_actions = [
        agent.select_action(state, epsilon=1.0)
        for _ in range(100)
    ]

    assert all(0 <= action < 3 for action in random_actions)
    assert len(set(random_actions)) > 1

    # Verify pure exploitation selects the maximum Q-value action.
    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state, device=device
        ).unsqueeze(0)

        q_values = agent.online_network(state_tensor)
        expected_action = q_values.argmax(dim=1).item()

    greedy_actions = [
        agent.select_action(state, epsilon=0.0)
        for _ in range(20)
    ]

    assert all(action == expected_action for action in greedy_actions)

    # Verify invalid epsilon values are rejected.
    for epsilon in (-0.1, 1.1):
        try:
            agent.select_action(state, epsilon)
            raise AssertionError("Invalid epsilon was accepted")
        except ValueError:
            pass

    print("Device:", device)
    print("Random actions sampled:", len(random_actions))
    print("Unique random actions:", sorted(set(random_actions)))
    print("Greedy action:", expected_action)
    print("Epsilon validation passed!")
    print("Epsilon-greedy test passed!")


if __name__ == "__main__":
    test_epsilon_greedy()
