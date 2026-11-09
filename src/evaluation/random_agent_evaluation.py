

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from src.utils.pong_environment import PongEnvironment


def evaluate_random_agent(
    num_episodes: int = 100,
    seed_start: int = 1000
) -> dict:
    """
    Evaluate a uniformly random agent on Atari Pong.

    Args:
        num_episodes: Number of evaluation episodes.
        seed_start: First seed in the evaluation sequence.

    Returns:
        Dictionary containing evaluation metrics and episode rewards.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    env = PongEnvironment(num_frames=4)
    rewards = []

    try:
        for episode in range(num_episodes):
            seed = seed_start + episode
            rng = random.Random(seed)

            env.reset(seed=seed)

            episode_reward = 0.0
            done = False

            while not done:
                action = rng.randrange(env.num_actions)

                _, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

    finally:
        env.close()

    rewards_array = np.asarray(rewards, dtype=np.float64)

    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "median_reward": float(np.median(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "win_rate": float(np.mean(rewards_array > 0) * 100),
        "episode_rewards": rewards
    }


def main() -> None:
    """Evaluate the random agent and save its results."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    results = evaluate_random_agent(
        num_episodes=args.episodes,
        seed_start=args.seed_start
    )

    output_path = Path(args.output_dir) / "random_agent"
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.csv"
    rewards_path = output_path / "episode_rewards.csv"

    metrics = {
        "agent": "Random",
        "episodes": args.episodes,
        "mean_reward": results["mean_reward"],
        "std_reward": results["std_reward"],
        "median_reward": results["median_reward"],
        "min_reward": results["min_reward"],
        "max_reward": results["max_reward"],
        "win_rate": results["win_rate"]
    }

    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metrics.keys()
        )

        writer.writeheader()
        writer.writerow(metrics)

    with rewards_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "seed", "reward"]
        )

        writer.writeheader()

        for episode, reward in enumerate(
            results["episode_rewards"]
        ):
            writer.writerow({
                "episode": episode + 1,
                "seed": args.seed_start + episode,
                "reward": reward
            })

    print("\nRandom agent evaluation completed!")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Episode rewards saved to: {rewards_path}")


if __name__ == "__main__":
    main()

