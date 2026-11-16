
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.training.dqn_agent import DQNAgent
from src.utils.ball_detector import detect_ball_x
from src.utils.pong_environment import PongEnvironment
from src.utils.rally_detector import RallyDetector


def test_rally_detection(
    checkpoint_path: str,
    output_dir: str,
    seed: int = 1000,
    max_steps: int = 10000
) -> None:
    """Test ball tracking and rally detection during Pong gameplay."""
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    agent = DQNAgent(
        num_actions=3,
        device=device,
        activation=checkpoint.get("activation", "leaky_relu"),
        double_dqn=checkpoint.get("double_dqn", False)
    )

    agent.online_network.load_state_dict(
        checkpoint["online_network"]
    )
    agent.online_network.eval()

    output_path = Path(output_dir)
    frames_path = output_path / "frames"
    frames_path.mkdir(parents=True, exist_ok=True)

    detector = RallyDetector()
    env = PongEnvironment(num_frames=4)

    rows = []
    hits = 0
    detections = 0

    try:
        state = env.reset(seed=seed)
        detector.reset()

        for step in range(1, max_steps + 1):
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state,
                    device=device
                ).unsqueeze(0)

                action = int(
                    agent.online_network(
                        state_tensor
                    ).argmax(dim=1).item()
                )

            state, reward, terminated, truncated, _ = env.step(action)

            frame = env.env.render()
            ball_x = detect_ball_x(frame)
            hit = detector.update(ball_x)

            if ball_x is not None:
                detections += 1

            if hit:
                hits += 1

            rows.append({
                "step": step,
                "ball_x": ball_x,
                "hit": hit,
                "reward": reward,
                "action": action
            })

            if hit or step % 500 == 0:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)

                if ball_x is not None:
                    x = round(ball_x)
                    draw.line(
                        [(x, 0), (x, image.height - 1)],
                        fill="red",
                        width=1
                    )

                if hit:
                    draw.rectangle(
                        [(0, 0), (image.width - 1, image.height - 1)],
                        outline="yellow",
                        width=3
                    )

                image.save(
                    frames_path / f"frame_{step:06d}.png"
                )

            if reward != 0:
                detector.reset()

            if terminated or truncated:
                break

    finally:
        env.close()

    csv_path = output_path / "rally_detection.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "ball_x", "hit", "reward", "action"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Steps:", len(rows))
    print("Ball detections:", detections)
    print("Detection rate:", f"{detections / len(rows) * 100:.2f}%")
    print("Potential paddle hits:", hits)
    print("Results:", csv_path)
    print("Frames:", frames_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10000)

    args = parser.parse_args()

    test_rally_detection(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps
    )
