
import argparse
import csv
import gc
from pathlib import Path

import torch

from src.evaluation.evaluate import evaluate_agent
from src.training.dqn_agent import DQNAgent


def evaluate_checkpoints(
    checkpoint_paths: list[str],
    output_dir: str = "experiments/results",
    num_episodes: int = 100,
    epsilon: float = 0.01,
    seed_start: int = 1000
) -> None:
    """
    Evaluate DQN checkpoints from the same experiment and save results.

    Args:
        checkpoint_paths: Paths to the checkpoints being evaluated.
        output_dir: Root directory for experiment evaluation results.
        num_episodes: Number of episodes per checkpoint.
        epsilon: Exploration probability during evaluation.
        seed_start: First seed in the evaluation sequence.
    """
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1.")

    paths = [Path(path) for path in checkpoint_paths]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

    experiment_names = {
        path.parent.name
        for path in paths
    }

    if len(experiment_names) != 1:
        raise ValueError(
            "All checkpoints must belong to the same experiment."
        )

    experiment_name = experiment_names.pop()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_path = output_path / "metrics.csv"
    rewards_path = output_path / "episode_rewards.csv"

    episode_seeds = list(
        range(seed_start, seed_start + num_episodes)
    )

    metrics_rows = []
    rewards_rows = []

    for path in paths:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True
        )

        agent = DQNAgent(
            num_actions=3,
            device=device
        )

        agent.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        agent.online_network.eval()

        step = int(checkpoint["step"])

        del checkpoint
        gc.collect()

        print(f"\nExperiment: {experiment_name}")
        print(f"Checkpoint step: {step:,}")
        print(f"Device: {device}")
        print(f"Episodes: {num_episodes}")

        results = evaluate_agent(
            agent=agent,
            num_episodes=num_episodes,
            epsilon=epsilon,
            episode_seeds=episode_seeds
        )

        metrics_rows.append({
            "experiment": experiment_name,
            "step": step,
            "episodes": num_episodes,
            "epsilon": epsilon,
            "mean_reward": results["mean_reward"],
            "std_reward": results["std_reward"],
            "median_reward": results["median_reward"],
            "min_reward": results["min_reward"],
            "max_reward": results["max_reward"],
            "win_rate": results["win_rate"]
        })

        for episode, reward in enumerate(
            results["episode_rewards"]
        ):
            rewards_rows.append({
                "experiment": experiment_name,
                "step": step,
                "episode": episode + 1,
                "seed": episode_seeds[episode],
                "reward": reward
            })

        print(f"Mean reward: {results['mean_reward']:.2f}")
        print(f"Std reward: {results['std_reward']:.2f}")
        print(f"Median reward: {results['median_reward']:.2f}")
        print(f"Min reward: {results['min_reward']:.2f}")
        print(f"Max reward: {results['max_reward']:.2f}")
        print(f"Win rate: {results['win_rate']:.2f}%")

        del agent
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "step",
                "episodes",
                "epsilon",
                "mean_reward",
                "std_reward",
                "median_reward",
                "min_reward",
                "max_reward",
                "win_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics_rows)

    with rewards_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "step",
                "episode",
                "seed",
                "reward"
            ]
        )

        writer.writeheader()
        writer.writerows(rewards_rows)

    print("\nFinal evaluation completed!")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Episode rewards saved to: {rewards_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    evaluate_checkpoints(
        checkpoint_paths=args.checkpoints,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        seed_start=args.seed_start
    )
