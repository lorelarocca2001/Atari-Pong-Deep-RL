
import argparse
import gc
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def record_gameplay(
    agent_type: str = "dqn",
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    seed: int = 1000,
    fps: int = 30
) -> None:
    """
    Record a Pong episode played by a DQN or random agent.

    Args:
        agent_type: Agent type, either dqn or random.
        checkpoint_path: DQN checkpoint path, required for dqn.
        output_path: Optional destination MP4 path.
        seed: Environment and random agent seed.
        fps: Output video frame rate.
    """
    if agent_type not in ("dqn", "random"):
        raise ValueError("Agent type must be dqn or random.")

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    agent = None
    rng = random.Random(seed)

    if agent_type == "dqn":
        if checkpoint_path is None:
            raise ValueError("A checkpoint is required for the DQN agent.")

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        experiment_name = checkpoint_path.parent.name

        checkpoint = torch.load(
            checkpoint_path,
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

        del checkpoint
        gc.collect()

    else:
        experiment_name = "random_agent"

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / experiment_name
            / "gameplay.mp4"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = PongEnvironment(num_frames=4)
    writer = None

    try:
        state = env.reset(seed=seed)
        frame = np.asarray(env.env.render())

        height, width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise RuntimeError("Could not initialize video writer.")

        writer.write(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

        episode_reward = 0.0
        total_steps = 0
        done = False

        while not done:
            if agent_type == "random":
                action = rng.randrange(env.num_actions)
            else:
                with torch.no_grad():
                    action = agent.select_action(
                        state,
                        epsilon=0.0
                    )

            state, reward, terminated, truncated, _ = env.step(
                action
            )

            frame = np.asarray(env.env.render())

            writer.write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )

            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

        print("\nGameplay recording completed!")
        print(f"Agent: {agent_type}")
        print(f"Seed: {seed}")
        print(f"Total steps: {total_steps}")
        print(f"Episode reward: {episode_reward}")
        print(f"Video saved to: {output_path}")

    finally:
        if writer is not None:
            writer.release()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "random"],
        default="dqn"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30
    )

    args = parser.parse_args()

    record_gameplay(
        agent_type=args.agent,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        fps=args.fps
    )

