
import argparse
import csv
import gc
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_game_positions(frame: np.ndarray) -> tuple | None:
    """Detect the ball and the agent paddle in an RGB Pong frame."""
    height, width = frame.shape[:2]

    rgb = frame.astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    paddle_mask = (
        (green > red + 25)
        & (green > blue + 25)
        & (green > 80)
    ).astype(np.uint8)

    paddle_mask[:, :int(width * 0.65)] = 0
    paddle_mask[:int(height * 0.12), :] = 0

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        paddle_mask, connectivity=8
    )

    paddle_candidates = [
        i for i in range(1, count)
        if 8 <= stats[i, cv2.CC_STAT_HEIGHT] <= 50
        and 1 <= stats[i, cv2.CC_STAT_WIDTH] <= 12
    ]

    if not paddle_candidates:
        return None

    paddle_index = max(
        paddle_candidates,
        key=lambda i: stats[i, cv2.CC_STAT_AREA]
    )

    paddle_y = float(centroids[paddle_index, 1])

    ball_mask = (
        (rgb.min(axis=2) >= 190)
        & ((rgb.max(axis=2) - rgb.min(axis=2)) <= 15)
    ).astype(np.uint8)

    ball_mask[:, :int(width * 0.08)] = 0
    ball_mask[:, int(width * 0.92):] = 0
    ball_mask[:int(height * 0.18), :] = 0
    ball_mask[int(height * 0.92):, :] = 0

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        ball_mask, connectivity=8
    )

    ball_candidates = [
        i for i in range(1, count)
        if 2 <= stats[i, cv2.CC_STAT_AREA] <= 80
        and stats[i, cv2.CC_STAT_WIDTH] <= 10
        and stats[i, cv2.CC_STAT_HEIGHT] <= 10
    ]

    if not ball_candidates:
        return None

    ball_index = max(
        ball_candidates,
        key=lambda i: stats[i, cv2.CC_STAT_AREA]
    )

    ball_x, ball_y = centroids[ball_index]

    relative_y = (ball_y - paddle_y) / height
    normalized_x = ball_x / width

    return float(relative_y), float(normalized_x)


def collect_policy_data(
    agent: DQNAgent,
    num_episodes: int,
    seed_start: int,
    gamma: float
) -> list[dict]:
    """Collect Q-values, TD errors, and positions from greedy Pong episodes."""
    env = PongEnvironment(num_frames=4)
    records = []

    agent.online_network.eval()
    agent.target_network.eval()

    try:
        for episode in range(num_episodes):
            state = env.reset(seed=seed_start + episode)
            done = False
            step = 0
            episode_reward = 0.0

            while not done:
                frame = np.asarray(env.env.render())
                positions = extract_game_positions(frame)

                state_tensor = torch.as_tensor(
                    state,
                    device=agent.device
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = agent.online_network(state_tensor)
                    action = int(q_values.argmax(dim=1).item())
                    selected_q = float(q_values[0, action].item())

                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )

                done = terminated or truncated

                next_state_tensor = torch.as_tensor(
                    next_state,
                    device=agent.device
                ).unsqueeze(0)

                with torch.no_grad():
                    next_q_values = agent.target_network(
                        next_state_tensor
                    )

                    next_q_max = float(
                        next_q_values.max(dim=1).values.item()
                    )

                target = reward + gamma * (1 - int(done)) * next_q_max
                td_error = target - selected_q

                records.append({
                    "episode": episode + 1,
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "q_noop": float(q_values[0, 0].item()),
                    "q_up": float(q_values[0, 1].item()),
                    "q_down": float(q_values[0, 2].item()),
                    "selected_q": selected_q,
                    "td_error": td_error,
                    "absolute_td_error": abs(td_error),
                    "relative_ball_y": (
                        positions[0] if positions is not None else None
                    ),
                    "ball_x": (
                        positions[1] if positions is not None else None
                    )
                })

                state = next_state
                episode_reward += reward
                step += 1

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Steps: {step}"
            )

    finally:
        env.close()

    return records


def save_records(records: list[dict], output_path: Path) -> None:
    """Save policy observations as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


def plot_td_errors(
    records: list[dict],
    figure_dir: Path
) -> None:
    """Plot signed and absolute TD error distributions."""
    td_errors = np.array([
        row["td_error"] for row in records
    ])

    plt.figure(figsize=(9, 5))
    plt.hist(td_errors, bins=60)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("TD error")
    plt.ylabel("Frequency")
    plt.title("TD Error Distribution")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "td_error_distribution.png",
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(np.abs(td_errors), bins=60)
    plt.xlabel("Absolute TD error")
    plt.ylabel("Frequency")
    plt.title("Absolute TD Error Distribution")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "absolute_td_error_distribution.png",
        dpi=300
    )
    plt.close()


def plot_q_values(
    records: list[dict],
    figure_dir: Path
) -> None:
    """Plot Q-value estimates during the first episode."""
    episode_records = [
        row for row in records if row["episode"] == 1
    ]

    steps = [row["step"] for row in episode_records]

    plt.figure(figsize=(12, 5))

    for key, label in [
        ("q_noop", "NOOP"),
        ("q_up", "UP"),
        ("q_down", "DOWN")
    ]:
        values = [row[key] for row in episode_records]
        plt.plot(steps, values, label=label, linewidth=1)

    plt.xlabel("Environment step")
    plt.ylabel("Estimated Q-value")
    plt.title("Q-Value Estimates During Gameplay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_dir / "q_values_over_time.png",
        dpi=300
    )
    plt.close()


def plot_q_advantages(
    records: list[dict],
    figure_dir: Path
) -> None:
    """Plot action advantages during the first episode."""
    episode_records = [
        row for row in records if row["episode"] == 1
    ]

    steps = np.array([
        row["step"] for row in episode_records
    ])

    q_values = np.array([
        [row["q_noop"], row["q_up"], row["q_down"]]
        for row in episode_records
    ])

    advantages = q_values - q_values.mean(
        axis=1,
        keepdims=True
    )

    plt.figure(figsize=(12, 5))

    for action, label in enumerate(["NOOP", "UP", "DOWN"]):
        plt.plot(
            steps,
            advantages[:, action],
            label=label,
            linewidth=1
        )

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    plt.xlabel("Environment step")
    plt.ylabel("Q-value advantage")
    plt.title("Action Advantages During Gameplay")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir / "q_value_advantages.png",
        dpi=300
    )

    plt.close()


def plot_policy_heatmap(
    records: list[dict],
    figure_dir: Path
) -> None:
    """Plot the most frequent DQN action across Pong state regions."""
    valid_records = [
        row for row in records
        if row["relative_ball_y"] is not None
        and row["ball_x"] is not None
    ]

    if not valid_records:
        raise ValueError("No valid ball and paddle positions detected.")

    relative_y = np.array([
        row["relative_ball_y"] for row in valid_records
    ])

    ball_x = np.array([
        row["ball_x"] for row in valid_records
    ])

    actions = np.array([
        row["action"] for row in valid_records
    ])

    x_edges = np.linspace(-1.0, 1.0, 21)
    y_edges = np.linspace(0.0, 1.0, 13)

    counts = np.zeros(
        (len(y_edges) - 1, len(x_edges) - 1, 3),
        dtype=np.int64
    )

    x_bins = np.searchsorted(x_edges, relative_y, side="right") - 1
    y_bins = np.searchsorted(y_edges, ball_x, side="right") - 1

    for x_bin, y_bin, action in zip(x_bins, y_bins, actions):
        if (
            0 <= x_bin < counts.shape[1]
            and 0 <= y_bin < counts.shape[0]
        ):
            counts[y_bin, x_bin, action] += 1

    totals = counts.sum(axis=2)
    dominant_actions = counts.argmax(axis=2).astype(float)

    dominant_actions[totals < 10] = np.nan

    cmap = ListedColormap([
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c"
    ])
    cmap.set_bad("#dddddd")

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        cmap.N
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(
        dominant_actions,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            x_edges[0],
            x_edges[-1],
            y_edges[0],
            y_edges[-1]
        ],
        cmap=cmap,
        norm=norm
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1, 2]
    )

    colorbar.ax.set_yticklabels([
        "NOOP",
        "UP",
        "DOWN"
    ])

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Relative ball height (ball_y - paddle_y) / screen_height"
    )
    ax.set_ylabel("Normalized ball X position")
    ax.set_title("DQN Policy Heatmap")

    fig.tight_layout()

    fig.savefig(
        figure_dir / "policy_heatmap.png",
        dpi=300
    )

    plt.close(fig)

    print(
        f"Object detection: {len(valid_records)}/{len(records)} "
        f"frames ({len(valid_records) / len(records) * 100:.1f}%)"
    )
    print(
        f"Heatmap cells with at least 10 samples: "
        f"{np.sum(totals >= 10)}/{totals.size}"
    )


def visualize_policy(
    checkpoint_path: str,
    num_episodes: int = 10,
    seed_start: int = 1000,
    gamma: float = 0.99,
    figures_root: str | None = None,
    results_root: str | None = None
) -> None:
    """Analyze a trained DQN policy and save visualizations."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Gamma must be between 0 and 1.")

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    experiment_name = path.parent.name

    figures_dir = (
        Path(figures_root) if figures_root else PROJECT_ROOT / "figures"
    ) / experiment_name

    results_dir = (
    Path(results_root)
    if results_root
    else PROJECT_ROOT / "experiments" / "results"
) / experiment_name

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

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

    agent.target_network.load_state_dict(
        checkpoint["target_network"]
    )

    agent.online_network.eval()
    agent.target_network.eval()

    del checkpoint
    gc.collect()

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")

    records = collect_policy_data(
        agent=agent,
        num_episodes=num_episodes,
        seed_start=seed_start,
        gamma=gamma
    )

    save_records(
        records,
        results_dir / "policy_analysis.csv"
    )

    plot_td_errors(records, figures_dir)
    plot_q_values(records, figures_dir)
    plot_q_advantages(records, figures_dir)
    plot_policy_heatmap(records, figures_dir)

    errors = np.array([
        row["td_error"] for row in records
    ])

    print("\nPolicy analysis completed!")
    print(f"Transitions: {len(records)}")
    print(f"Mean TD error: {errors.mean():.4f}")
    print(f"Mean absolute TD error: {np.abs(errors).mean():.4f}")
    print(f"TD error std: {errors.std():.4f}")
    print(f"Figures: {figures_dir}")
    print(f"Results: {results_dir}")


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
        default=10
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99
    )

    parser.add_argument(
        "--figures-root",
        type=str,
        default=None
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default=None
    )

    args = parser.parse_args()

    visualize_policy(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        seed_start=args.seed_start,
        gamma=args.gamma,
        figures_root=args.figures_root,
        results_root=args.results_root
    )
