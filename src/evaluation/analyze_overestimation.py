
import argparse
import csv
import gc
import random
from pathlib import Path

import numpy as np
import torch

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


def analyze_checkpoint(
    checkpoint_path: str,
    output_dir: str = "experiments/results",
    num_episodes: int = 100,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    seed_start: int = 1000
) -> dict:
    """Compare predicted Q-values with observed discounted returns."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Invalid epsilon.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Invalid gamma.")

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    experiment_name = path.parent.name
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    activation = checkpoint.get("activation", "leaky_relu")
    double_dqn = checkpoint.get("double_dqn", False)
    step = int(checkpoint["step"])

    agent = DQNAgent(
        num_actions=3,
        device=device,
        activation=activation,
        double_dqn=double_dqn
    )

    agent.online_network.load_state_dict(
        checkpoint["online_network"]
    )
    agent.online_network.eval()

    del checkpoint
    gc.collect()

    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    transitions_path = output_path / "overestimation_transitions.csv"
    episodes_path = output_path / "overestimation_episodes.csv"
    metrics_path = output_path / "overestimation_metrics.csv"

    all_errors = []
    episode_rows = []
    env = PongEnvironment(num_frames=4)

    print(f"\nExperiment: {experiment_name}")
    print(f"Algorithm: {'Double DQN' if double_dqn else 'DQN'}")
    print(f"Checkpoint step: {step:,}")
    print(f"Device: {device}")

    try:
        with transitions_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "episode",
                "seed",
                "timestep",
                "action",
                "q_value",
                "discounted_return",
                "estimation_error"
            ])

            for episode in range(num_episodes):
                seed = seed_start + episode
                rng = random.Random(seed)

                state = env.reset(seed=seed)
                done = False

                q_predictions = []
                rewards = []
                actions = []

                with torch.no_grad():
                    while not done:
                        state_tensor = torch.as_tensor(
                            state,
                            device=device
                        ).unsqueeze(0)

                        q_values = agent.online_network(state_tensor)

                        if rng.random() < epsilon:
                            action = rng.randrange(agent.num_actions)
                        else:
                            action = int(q_values.argmax(dim=1).item())

                        predicted_q = float(
                            q_values[0, action].item()
                        )

                        next_state, reward, terminated, truncated, _ = (
                            env.step(action)
                        )

                        q_predictions.append(predicted_q)
                        rewards.append(float(reward))
                        actions.append(action)

                        state = next_state
                        done = terminated or truncated

                discounted_returns = np.empty(
                    len(rewards),
                    dtype=np.float64
                )

                running_return = 0.0

                for t in range(len(rewards) - 1, -1, -1):
                    running_return = (
                        rewards[t] + gamma * running_return
                    )
                    discounted_returns[t] = running_return

                errors = (
                    np.asarray(q_predictions, dtype=np.float64)
                    - discounted_returns
                )

                all_errors.extend(errors.tolist())

                for t, error in enumerate(errors):
                    writer.writerow([
                        episode + 1,
                        seed,
                        t,
                        actions[t],
                        q_predictions[t],
                        discounted_returns[t],
                        float(error)
                    ])

                episode_rows.append({
                    "episode": episode + 1,
                    "seed": seed,
                    "reward": float(sum(rewards)),
                    "mean_error": float(np.mean(errors)),
                    "mae": float(np.mean(np.abs(errors))),
                    "overestimation_rate": float(
                        np.mean(errors > 0) * 100
                    )
                })

                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward {sum(rewards):.1f} | "
                    f"Mean error {np.mean(errors):.4f}"
                )

    finally:
        env.close()

    errors_array = np.asarray(all_errors, dtype=np.float64)

    metrics = {
        "experiment": experiment_name,
        "algorithm": "Double DQN" if double_dqn else "DQN",
        "step": step,
        "episodes": num_episodes,
        "epsilon": epsilon,
        "gamma": gamma,
        "transitions": len(errors_array),
        "mean_error": float(np.mean(errors_array)),
        "mae": float(np.mean(np.abs(errors_array))),
        "rmse": float(np.sqrt(np.mean(errors_array ** 2))),
        "overestimation_rate": float(
            np.mean(errors_array > 0) * 100
        )
    }

    with episodes_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(episode_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(metrics.keys())
        )
        writer.writeheader()
        writer.writerow(metrics)

    print("\nOverestimation analysis completed!")
    print(f"Mean error: {metrics['mean_error']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(
        f"Overestimation rate: "
        f"{metrics['overestimation_rate']:.2f}%"
    )
    print(f"Metrics saved to: {metrics_path}")

    del agent
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


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
        "--gamma",
        type=float,
        default=0.99
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    for checkpoint_path in args.checkpoints:
        analyze_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            epsilon=args.epsilon,
            gamma=args.gamma,
            seed_start=args.seed_start
        )
