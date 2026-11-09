
import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import torch

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ["NOOP", "UP", "DOWN"]
COLORS = [(255, 160, 70), (70, 180, 255), (100, 220, 120)]


def compute_occlusion_saliency(
    network: torch.nn.Module,
    state: np.ndarray,
    action: int,
    original_q: float,
    device: torch.device,
    patch_size: int = 8,
    stride: int = 8,
    batch_size: int = 64
) -> np.ndarray:
    """Compute occlusion saliency on the newest stacked frame."""
    height, width = state.shape[-2:]

    original = state[-1]
    blurred = cv2.GaussianBlur(original, (11, 11), 0)

    centers = [
        (y, x)
        for y in range(stride // 2, height, stride)
        for x in range(stride // 2, width, stride)
    ]

    saliency = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(centers), batch_size):
        batch_centers = centers[start:start + batch_size]

        batch_states = np.repeat(
            state[None, :, :, :],
            len(batch_centers),
            axis=0
        )

        masks = []

        for index, (cy, cx) in enumerate(batch_centers):
            radius = patch_size // 2

            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius)
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius)

            yy, xx = np.mgrid[y0:y1, x0:x1]

            sigma = max(patch_size / 3, 1)

            mask = np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2)
                / (2 * sigma ** 2)
            ).astype(np.float32)

            patch = original[y0:y1, x0:x1].astype(np.float32)
            blurred_patch = blurred[y0:y1, x0:x1].astype(
                np.float32
            )

            perturbed = (
                patch * (1 - mask)
                + blurred_patch * mask
            )

            batch_states[index, -1, y0:y1, x0:x1] = (
                perturbed.astype(np.uint8)
            )

            masks.append((y0, y1, x0, x1, mask))

        batch_tensor = torch.as_tensor(
            batch_states,
            device=device
        )

        with torch.inference_mode():
            q_values = network(batch_tensor)
            perturbed_q = q_values[:, action].cpu().numpy()

        differences = np.maximum(
            original_q - perturbed_q,
            0.0
        )

        for difference, mask_data in zip(differences, masks):
            y0, y1, x0, x1, mask = mask_data

            saliency[y0:y1, x0:x1] += difference * mask
            weights[y0:y1, x0:x1] += mask

    saliency = saliency / np.maximum(weights, 1e-8)

    return saliency


def create_saliency_overlay(
    state: np.ndarray,
    saliency: np.ndarray
) -> np.ndarray:
    """Overlay the saliency map on the latest grayscale frame."""
    frame = state[-1]

    gray = cv2.resize(
        frame,
        (320, 320),
        interpolation=cv2.INTER_NEAREST
    )

    heatmap = cv2.resize(
        saliency,
        (320, 320),
        interpolation=cv2.INTER_LINEAR
    )

    scale = float(np.percentile(heatmap, 99))

    if scale > 1e-8:
        normalized = np.clip(heatmap / scale, 0, 1)
    else:
        normalized = np.zeros_like(heatmap)

    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    alpha = (normalized * 0.75)[..., None]

    overlay = (
        gray_bgr.astype(np.float32) * (1 - alpha)
        + colored.astype(np.float32) * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_text(
    canvas: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1
) -> None:
    """Draw text on the visualization canvas."""
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def create_xray_frame(
    rgb_frame: np.ndarray,
    state: np.ndarray,
    saliency: np.ndarray,
    q_values: np.ndarray,
    action: int,
    step: int,
    episode_reward: float,
    saliency_step: int
) -> np.ndarray:
    """Compose gameplay, saliency, and DQN decision panels."""
    canvas = np.full(
        (560, 1100, 3),
        (22, 24, 30),
        dtype=np.uint8
    )

    draw_text(
        canvas,
        "DQN X-RAY GAMEPLAY",
        (25, 40),
        scale=0.95,
        thickness=2
    )

    draw_text(canvas, "GAMEPLAY", (25, 75))
    draw_text(canvas, "NETWORK VISION", (390, 75))
    draw_text(canvas, "DQN DECISIONS", (760, 75))

    game_frame = cv2.cvtColor(
        rgb_frame,
        cv2.COLOR_RGB2BGR
    )

    game_frame = cv2.resize(
        game_frame,
        (320, 420),
        interpolation=cv2.INTER_NEAREST
    )

    canvas[90:510, 25:345] = game_frame

    overlay = create_saliency_overlay(state, saliency)

    canvas[90:410, 390:710] = overlay

    cv2.rectangle(
        canvas,
        (390, 90),
        (709, 409),
        (95, 95, 95),
        1
    )

    draw_text(
        canvas,
        "Frame: newest of 4",
        (390, 445),
        scale=0.5
    )

    draw_text(
        canvas,
        f"Saliency updated at step {saliency_step}",
        (390, 470),
        scale=0.45
    )

    draw_text(
        canvas,
        "Occlusion: positive Q-value drop",
        (390, 495),
        scale=0.44
    )

    draw_text(
        canvas,
        f"Selected action: {ACTIONS[action]}",
        (760, 115),
        scale=0.65,
        color=(120, 230, 130),
        thickness=2
    )

    q_min = min(float(np.min(q_values)), 0.0)
    q_max = max(float(np.max(q_values)), 0.0)

    span = max(q_max - q_min, 0.1)
    margin = span * 0.15
    q_min -= margin
    q_max += margin
    span = q_max - q_min

    bar_left = 765
    bar_width = 295
    zero_x = int(
        bar_left + (0 - q_min) / span * bar_width
    )

    for index, label in enumerate(ACTIONS):
        y = 165 + index * 87

        selected = index == action

        color = (
            COLORS[index]
            if selected
            else (105, 110, 120)
        )

        draw_text(
            canvas,
            f"{label}   Q = {q_values[index]:+.4f}",
            (765, y),
            scale=0.56,
            color=color,
            thickness=2 if selected else 1
        )

        cv2.rectangle(
            canvas,
            (bar_left, y + 15),
            (bar_left + bar_width, y + 36),
            (48, 52, 60),
            -1
        )

        value_x = int(
            bar_left
            + (float(q_values[index]) - q_min)
            / span * bar_width
        )

        cv2.rectangle(
            canvas,
            (min(zero_x, value_x), y + 15),
            (max(zero_x, value_x), y + 36),
            color,
            -1
        )

        cv2.line(
            canvas,
            (zero_x, y + 12),
            (zero_x, y + 39),
            (210, 210, 210),
            1
        )

    draw_text(
        canvas,
        f"Max Q (state value): {np.max(q_values):+.4f}",
        (760, 445),
        scale=0.55
    )

    draw_text(
        canvas,
        f"Episode reward: {episode_reward:+.0f}",
        (760, 475),
        scale=0.55
    )

    draw_text(
        canvas,
        f"Environment step: {step}",
        (760, 505),
        scale=0.55
    )

    draw_text(
        canvas,
        "Atari Pong | Greedy DQN | Occlusion Saliency",
        (25, 540),
        scale=0.47,
        color=(155, 160, 170)
    )

    return canvas


def record_xray_gameplay(
    checkpoint_path: str,
    output_path: str | None = None,
    seed: int = 1000,
    fps: int = 30,
    saliency_frequency: int = 10,
    patch_size: int = 8,
    stride: int = 8,
    batch_size: int = 64,
    max_steps: int | None = None
) -> None:
    """Record DQN gameplay with occlusion saliency and Q-values."""
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    if saliency_frequency <= 0:
        raise ValueError("Saliency frequency must be positive.")

    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("Saliency parameters must be positive.")

    if max_steps is not None and max_steps <= 0:
        raise ValueError("Max steps must be positive.")

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    experiment_name = path.parent.name

    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / experiment_name
            / "xray_gameplay.mp4"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    agent.online_network.eval()

    del checkpoint
    gc.collect()

    env = PongEnvironment(num_frames=4)
    writer = None

    try:
        state = env.reset(seed=seed)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (1100, 560)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not initialize MP4 video writer."
            )

        total_steps = 0
        episode_reward = 0.0
        done = False

        saliency = np.zeros(
            (84, 84),
            dtype=np.float32
        )

        saliency_step = 0

        while not done:
            rgb_frame = np.asarray(
                env.env.render()
            ).copy()

            state_tensor = torch.as_tensor(
                state,
                device=device
            ).unsqueeze(0)

            with torch.inference_mode():
                q_tensor = agent.online_network(
                    state_tensor
                )

                q_values = (
                    q_tensor[0].cpu().numpy().copy()
                )

            action = int(np.argmax(q_values))

            if total_steps % saliency_frequency == 0:
                saliency = compute_occlusion_saliency(
                    network=agent.online_network,
                    state=state,
                    action=action,
                    original_q=float(q_values[action]),
                    device=device,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size
                )

                saliency_step = total_steps

            video_frame = create_xray_frame(
                rgb_frame=rgb_frame,
                state=state,
                saliency=saliency,
                q_values=q_values,
                action=action,
                step=total_steps,
                episode_reward=episode_reward,
                saliency_step=saliency_step
            )

            writer.write(video_frame)

            state, reward, terminated, truncated, _ = env.step(
                action
            )

            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

            if total_steps % 250 == 0:
                print(
                    f"Step: {total_steps} | "
                    f"Reward: {episode_reward:+.0f}"
                )

            if max_steps is not None and total_steps >= max_steps:
                break

        print("\nX-Ray gameplay recording completed!")
        print(f"Experiment: {experiment_name}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Steps: {total_steps}")
        print(f"Episode completed: {done}")
        print(f"Episode reward: {episode_reward:+.0f}")
        print(f"Video saved to: {output_path}")

    finally:
        if writer is not None:
            writer.release()

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
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

    parser.add_argument(
        "--saliency-frequency",
        type=int,
        default=10
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None
    )

    args = parser.parse_args()

    record_xray_gameplay(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        fps=args.fps,
        saliency_frequency=args.saliency_frequency,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps
    )
