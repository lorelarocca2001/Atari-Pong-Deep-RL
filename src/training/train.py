import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.datasets.replay_buffer import ReplayBuffer
from src.evaluation.evaluate import evaluate_agent
from src.training.dqn_agent import DQNAgent
from src.training.epsilon_scheduler import LinearEpsilonScheduler
from src.utils.config import load_config
from src.utils.pong_environment import PongEnvironment
from src.utils.random_state import get_random_state, set_random_state
from src.utils.seed import set_seed
from src.utils.wandb_logger import WandbLogger


def save_checkpoint(
    agent: DQNAgent,
    buffer: ReplayBuffer,
    env: PongEnvironment,
    step: int,
    directory: Path,
    state: np.ndarray,
    episode_reward: float,
    episode_count: int,
    episode_rewards: deque,
    last_loss: float
) -> None:
    """
    Save the complete DQN training state for resuming training.

    Args:
        agent: DQN agent containing networks and optimizer.
        buffer: Experience replay buffer.
        env: Atari Pong environment.
        step: Current environment step.
        directory: Destination directory.
        state: Current stacked observation.
        episode_reward: Accumulated reward of the current episode.
        episode_count: Number of completed episodes.
        episode_rewards: Rewards from the latest 100 episodes.
        last_loss: Most recent training loss.
    """
    directory.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "activation": agent.activation,
        "online_network": agent.online_network.state_dict(),
        "target_network": agent.target_network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "replay_buffer": buffer.state_dict(),
        "environment": env.state_dict(),
        "random_state": get_random_state(),
        "training_state": {
            "state": torch.from_numpy(state.copy()),
            "episode_reward": episode_reward,
            "episode_count": episode_count,
            "episode_rewards": list(episode_rewards),
            "last_loss": last_loss
        }
    }

    checkpoint_path = directory / f"dqn_step_{step}.pth"
    temporary_path = directory / f"dqn_step_{step}.pth.tmp"

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    agent: DQNAgent,
    buffer: ReplayBuffer,
    env: PongEnvironment,
    checkpoint_path: str,
    device: torch.device
) -> dict:
    """
    Restore the complete DQN training state from a checkpoint.

    Args:
        agent: DQN agent receiving the saved network and optimizer states.
        buffer: Replay buffer receiving the saved transitions.
        env: Pong environment receiving the saved emulator state.
        checkpoint_path: Path to the checkpoint file.
        device: Device used for neural network computations.

    Returns:
        Dictionary containing the restored training progress.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    checkpoint_activation = checkpoint.get(
    "activation",
    "leaky_relu"
    )
    
    if checkpoint_activation != agent.activation:
       raise ValueError(
        f"Checkpoint activation mismatch: "
        f"checkpoint uses '{checkpoint_activation}', "
        f"but the agent uses '{agent.activation}'."
    )

    agent.online_network.load_state_dict(
        checkpoint["online_network"]
    )

    agent.target_network.load_state_dict(
        checkpoint["target_network"]
    )

    agent.optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    agent.online_network.to(device)
    agent.target_network.to(device)
    agent.target_network.eval()

    buffer.load_state_dict(
        checkpoint["replay_buffer"]
    )

    env.load_state_dict(
        checkpoint["environment"]
    )

    training_state = checkpoint["training_state"]

    restored_state = {
        "step": int(checkpoint["step"]),
        "state": training_state["state"].cpu().numpy().copy(),
        "episode_reward": float(training_state["episode_reward"]),
        "episode_count": int(training_state["episode_count"]),
        "episode_rewards": deque(
            training_state["episode_rewards"],
            maxlen=100
        ),
        "last_loss": float(training_state["last_loss"])
    }

    set_random_state(checkpoint["random_state"])

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Restored step: {restored_state['step']}")
    print(f"Restored transitions: {len(buffer)}")
    print(f"Restored episodes: {restored_state['episode_count']}")

    return restored_state


def train(
    config_path: str = "experiments/configs/dqn.yaml",
    resume_from: str | None = None
) -> None:
    """
    Train a DQN agent or resume training from a complete checkpoint.

    Args:
        config_path: Path to the experiment configuration file.
        resume_from: Optional path to a complete training checkpoint.
    """
    config = load_config(config_path)
    seed = config["seed"]

    set_seed(seed)

    experiment_name = Path(config_path).stem

    if resume_from is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{experiment_name}"
    else:
        checkpoint_path = Path(resume_from).resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        run_name = checkpoint_path.parent.name

        if not run_name.endswith(f"_{experiment_name}"):
            raise ValueError(
                "Checkpoint experiment name does not match YAML configuration."
            )

    training_config = config["training"]
    replay_config = config["replay_buffer"]
    target_config = config["target_network"]
    optimization_config = config["optimization"]
    exploration_config = config["exploration"]
    evaluation_config = config["evaluation"]
    logging_config = config["logging"]
    checkpoint_config = config["checkpoint"]

    total_steps = training_config["total_steps"]
    batch_size = training_config["batch_size"]
    learning_rate = training_config["learning_rate"]
    gamma = training_config["gamma"]
    activation = config.get("model", {}).get("activation", "leaky_relu")

    buffer_capacity = replay_config["capacity"]
    warmup_steps = replay_config["warmup_steps"]

    train_frequency = optimization_config["train_frequency"]
    target_update_frequency = target_config["update_frequency"]

    epsilon_start = exploration_config["epsilon_start"]
    epsilon_end = exploration_config["epsilon_end"]
    epsilon_decay_steps = exploration_config["epsilon_decay_steps"]

    evaluation_frequency = evaluation_config["frequency"]
    evaluation_episodes = evaluation_config["episodes"]
    evaluation_epsilon = evaluation_config["epsilon"]

    log_frequency = logging_config["frequency"]
    log_dir = Path(logging_config["directory"]) / run_name

    checkpoint_frequency = checkpoint_config["frequency"]
    checkpoint_dir = Path(checkpoint_config["directory"]) / run_name

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    env = PongEnvironment(num_frames=4)

    agent = DQNAgent(
        num_actions=env.num_actions,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
        activation=activation
    )

    buffer = ReplayBuffer(capacity=buffer_capacity)

    scheduler = LinearEpsilonScheduler(
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        decay_steps=epsilon_decay_steps
    )

    logger = None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = deque(maxlen=100)
    episode_reward = 0.0
    episode_count = 0
    last_loss = float("nan")
    start_step = 0

    state = env.reset(seed=seed if resume_from is None else None)

    try:
        if resume_from is not None:
            restored = load_checkpoint(
                agent=agent,
                buffer=buffer,
                env=env,
                checkpoint_path=str(checkpoint_path),
                device=device
            )

            start_step = restored["step"]
            state = restored["state"]
            episode_reward = restored["episode_reward"]
            episode_count = restored["episode_count"]
            episode_rewards = restored["episode_rewards"]
            last_loss = restored["last_loss"]

            if start_step >= total_steps:
                raise ValueError(
                    "Checkpoint step must be lower than total_steps."
                )

        metrics_path = log_dir / "training_metrics.csv"

        if resume_from is not None:
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Training metrics file not found: {metrics_path}"
                )

            with metrics_path.open("r", newline="") as existing_file:
                rows = csv.DictReader(existing_file)
                latest_logged_step = max(
                    (int(row["step"]) for row in rows),
                    default=0
                )

            if latest_logged_step > start_step:
                raise RuntimeError(
                    "CSV contains data newer than the checkpoint. "
                    "Resume was stopped to prevent mixing timelines."
                )
        elif metrics_path.exists():
            raise FileExistsError(
                f"Training metrics file already exists: {metrics_path}"
            )
        logger = WandbLogger(
            project=logging_config["wandb_project"],
            run_name=run_name,
            config=config,
            log_dir=str(log_dir),
            resume=resume_from is not None
        )

        print("Experiment:", run_name)
        print("Seed:", seed)
        print("Device:", device)
        print("Activation:", activation)
        print("Total steps:", total_steps)
        print("Checkpoint directory:", checkpoint_dir)
        print("Log directory:", log_dir)
        print("Training started!")

        with metrics_path.open(
            "a" if resume_from is not None else "w",
            newline=""
        ) as log_file:
            writer = csv.writer(log_file)

            if resume_from is None:
                writer.writerow(
                    ["step", "episode", "reward", "average_reward", "epsilon"]
                )

            for step in range(start_step + 1, total_steps + 1):
                epsilon = scheduler.get_epsilon(step - 1)
                action = agent.select_action(state, epsilon)

                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )

                done = terminated or truncated

                buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                episode_reward += reward

                metrics = {}

                if done:
                    episode_count += 1
                    episode_rewards.append(episode_reward)

                    average_reward = float(np.mean(episode_rewards))

                    metrics.update({
                        "episode/reward": episode_reward,
                        "episode/average_reward_100": average_reward,
                        "episode/count": episode_count
                    })

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

                if step % log_frequency == 0:
                    metrics.update({
                        "train/epsilon": epsilon,
                        "train/replay_buffer_size": len(buffer)
                    })

                    if np.isfinite(last_loss):
                        metrics["train/td_loss"] = last_loss

                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            state,
                            device=device
                        ).unsqueeze(0)

                        q_values = agent.online_network(state_tensor)

                        metrics["train/mean_q_value"] = (
                            q_values.mean().item()
                        )

                    print(
                        f"Step {step}/{total_steps} | "
                        f"Buffer {len(buffer)} | "
                        f"Loss {last_loss:.5f} | "
                        f"Epsilon {epsilon:.4f}"
                    )

                if step % evaluation_frequency == 0:
                    results = evaluate_agent(
                        agent=agent,
                        num_episodes=evaluation_episodes,
                        epsilon=evaluation_epsilon
                    )

                    metrics.update({
                        "evaluation/mean_reward": results["mean_reward"],
                        "evaluation/std_reward": results["std_reward"]
                    })

                    print(
                        f"Evaluation at step {step} | "
                        f"Mean Reward {results['mean_reward']:.2f} | "
                        f"Std {results['std_reward']:.2f}"
                    )

                if metrics:
                    logger.log(metrics, step=step)

                if step % checkpoint_frequency == 0:
                    log_file.flush()

                    save_checkpoint(
                        agent=agent,
                        buffer=buffer,
                        env=env,
                        step=step,
                        directory=checkpoint_dir,
                        state=state,
                        episode_reward=episode_reward,
                        episode_count=episode_count,
                        episode_rewards=episode_rewards,
                        last_loss=last_loss
                    )

        if total_steps % checkpoint_frequency != 0:
            save_checkpoint(
                agent=agent,
                buffer=buffer,
                env=env,
                step=total_steps,
                directory=checkpoint_dir,
                state=state,
                episode_reward=episode_reward,
                episode_count=episode_count,
                episode_rewards=episode_rewards,
                last_loss=last_loss
            )

        print("Training completed!")
        print("Episodes completed:", episode_count)

    finally:
        env.close()

        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/dqn.yaml"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_from=args.resume_from
    )



