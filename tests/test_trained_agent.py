
import argparse
from collections import Counter

import numpy as np
import torch

from src.models.dqn import DQN
from src.utils.pong_environment import PongEnvironment


def test_trained_agent(checkpoint_path: str, num_episodes: int = 3) -> None:
    """
    Evaluate a trained DQN and analyze its action distribution and Q-values.

    Args:
        checkpoint_path: Path to the saved training checkpoint.
        num_episodes: Number of episodes used for diagnostics.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = DQN(num_actions=3).to(device)
    model.load_state_dict(checkpoint["online_network"])
    model.eval()

    environment = PongEnvironment(num_frames=4)

    action_counts = Counter()
    q_values_sum = np.zeros(3, dtype=np.float64)
    episode_rewards = []
    total_actions = 0

    try:
        with torch.no_grad():
            for episode in range(num_episodes):
                state = environment.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    state_tensor = torch.as_tensor(
                        state,
                        dtype=torch.uint8,
                        device=device
                    ).unsqueeze(0)

                    q_values = model(state_tensor).squeeze(0)
                    action = int(torch.argmax(q_values).item())

                    action_counts[action] += 1
                    q_values_sum += q_values.cpu().numpy()
                    total_actions += 1

                    state, reward, terminated, truncated, _ = environment.step(
                        action
                    )

                    episode_reward += reward
                    done = terminated or truncated

                episode_rewards.append(episode_reward)

                print(
                    f"Episode {episode + 1} | "
                    f"Reward: {episode_reward:.1f}"
                )
    finally:
        environment.close()

    action_names = ["NOOP", "UP", "DOWN"]

    print(f"\nCheckpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Total actions: {total_actions}")

    print("\nAction distribution:")

    for action, name in enumerate(action_names):
        percentage = 100 * action_counts[action] / total_actions
        print(f"{name}: {action_counts[action]} ({percentage:.2f}%)")

    print("\nMean Q-values:")

    mean_q_values = q_values_sum / total_actions

    for name, value in zip(action_names, mean_q_values):
        print(f"{name}: {value:.6f}")

    print(f"\nMean reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3
    )

    args = parser.parse_args()

    test_trained_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes
    )
