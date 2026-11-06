
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.datasets.replay_buffer import ReplayBuffer
from src.training.dqn_agent import DQNAgent
from src.training.epsilon_scheduler import LinearEpsilonScheduler
from src.utils.pong_environment import PongEnvironment


def save_checkpoint(agent: DQNAgent, step: int, directory: Path) -> None:
    """
    Save DQN network weights and optimizer state.

    Args:
        agent: DQN agent to save.
        step: Current environment step.
        directory: Destination directory.
    """
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "online_network": agent.online_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "optimizer": agent.optimizer.state_dict()
        },
        directory / f"dqn_step_{step}.pth"
    )


def train(total_steps: int = 2000) -> None:
    """
    Train a DQN agent on Atari Pong.

    Args:
        total_steps: Total number of environment interactions.
    """
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 32
    buffer_capacity = 5000
    warmup_steps = 1000
    train_frequency = 4
    target_update_frequency = 1000
    checkpoint_frequency = 1000

    env = PongEnvironment(num_frames=4)

    agent = DQNAgent(
        num_actions=env.num_actions,
        device=device,
        learning_rate=1e-4,
        gamma=0.99
    )

    buffer = ReplayBuffer(capacity=buffer_capacity)

    scheduler = LinearEpsilonScheduler(
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_steps=1_000_000
    )

    checkpoint_dir = Path("experiments/checkpoints")
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    episode_reward = 0.0
    episode_count = 0
    last_loss = float("nan")

    state = env.reset()

    print("Device:", device)
    print("Total steps:", total_steps)
    print("Training started!")

    try:
        with (log_dir / "training_metrics.csv").open(
            "w", newline=""
        ) as log_file:
            writer = csv.writer(log_file)

            writer.writerow(
                ["step", "episode", "reward", "average_reward", "epsilon"]
            )

            for step in range(1, total_steps + 1):
                epsilon = scheduler.get_epsilon(step - 1)

                action = agent.select_action(state, epsilon)

                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )

                done = terminated or truncated

                buffer.push(
                    state, action, reward, next_state, done
                )

                episode_reward += reward

                if done:
                    episode_count += 1
                    episode_rewards.append(episode_reward)

                    average_reward = float(np.mean(episode_rewards))

                    writer.writerow(
                        [
                            step,
                            episode_count,
                            episode_reward,
                            average_reward,
                            epsilon
                        ]
                    )
                    log_file.flush()

                    print(
                        f"Episode {episode_count} | "
                        f"Step {step} | "
                        f"Reward {episode_reward:.1f} | "
                        f"Avg100 {average_reward:.2f} | "
                        f"Epsilon {epsilon:.4f}"
                    )

                    state = env.reset()
                    episode_reward = 0.0
                else:
                    state = next_state

                if (
                    step >= warmup_steps
                    and step % train_frequency == 0
                    and len(buffer) >= batch_size
                ):
                    batch = buffer.sample(batch_size, device)
                    last_loss = agent.train_step(batch)

                if step % target_update_frequency == 0:
                    agent.update_target_network()

                if step % checkpoint_frequency == 0:
                    save_checkpoint(agent, step, checkpoint_dir)

                if step % 200 == 0:
                    print(
                        f"Step {step}/{total_steps} | "
                        f"Buffer {len(buffer)} | "
                        f"Loss {last_loss:.5f} | "
                        f"Epsilon {epsilon:.4f}"
                    )

        save_checkpoint(agent, total_steps, checkpoint_dir)

        print("Training completed!")
        print("Episodes completed:", episode_count)

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=2000
    )

    args = parser.parse_args()

    train(total_steps=args.steps)
