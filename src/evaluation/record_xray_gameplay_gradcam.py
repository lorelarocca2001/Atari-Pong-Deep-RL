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


def compute_gradcam(
    network: torch.nn.Module,
    state: np.ndarray,
    action: int,
    device: torch.device,
    layer_name: str = "features.0"
) -> np.ndarray:
    """Compute action-specific Grad-CAM for a convolutional layer."""
    layers = dict(network.named_modules())

    if layer_name not in layers or not isinstance(layers[layer_name], torch.nn.Conv2d):
        raise ValueError(f"Convolutional layer not found: {layer_name}")

    captured = {}

    def forward_hook(module, inputs, output):
        captured["activations"] = output.detach()
        output.register_hook(
            lambda gradient: captured.update(gradients=gradient.detach())
        )

    handle = layers[layer_name].register_forward_hook(forward_hook)

    try:
        network.zero_grad(set_to_none=True)
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)

        with torch.enable_grad():
            q_values = network(state_tensor)
            q_values[0, action].backward()

        activations = captured["activations"]
        gradients = captured["gradients"]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1))[0]
        cam = cam.cpu().numpy()

        cam = cv2.resize(
            cam,
            (state.shape[2], state.shape[1]),
            interpolation=cv2.INTER_LINEAR
        )

        peak = float(cam.max())
        if peak > 1e-8:
            cam = cam / peak
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    finally:
        handle.remove()
        network.zero_grad(set_to_none=True)


def create_gradcam_overlay(
    state: np.ndarray,
    gradcam: np.ndarray
) -> np.ndarray:
    """Overlay Grad-CAM on the newest preprocessed frame."""
    gray = cv2.resize(
        state[-1],
        (320, 320),
        interpolation=cv2.INTER_NEAREST
    )

    heatmap = cv2.resize(
        gradcam,
        (320, 320),
        interpolation=cv2.INTER_LINEAR
    )
    heatmap = np.clip(heatmap, 0.0, 1.0)

    colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = (heatmap * 0.7)[..., None]

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
    gradcam: np.ndarray,
    q_values: np.ndarray,
    action: int,
    step: int,
    episode_reward: float,
    gradcam_step: int,
    gradcam_layer: str
) -> np.ndarray:
    """Compose gameplay, Grad-CAM, and DQN decision panels."""
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

    overlay = create_gradcam_overlay(state, gradcam)

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
        f"Grad-CAM: {gradcam_layer} (4-frame input)",
        (390, 445),
        scale=0.5
    )

    draw_text(
        canvas,
        f"Grad-CAM updated at step {gradcam_step}",
        (390, 470),
        scale=0.45
    )

    draw_text(
        canvas,
        "Grad-CAM: selected-action Q-value",
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
        "Atari Pong | Greedy DQN | Grad-CAM",
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
    gradcam_frequency: int = 5,
    gradcam_layer: str = "features.0",
    max_steps: int | None = None
) -> None:
    """Record DQN gameplay with Grad-CAM and action Q-values."""
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    if gradcam_frequency <= 0:
        raise ValueError("Grad-CAM frequency must be positive.")

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
            / "xray_gradcam_gameplay.mp4"
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
            (1920, 1080)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not initialize MP4 video writer."
            )

        total_steps = 0
        episode_reward = 0.0
        done = False

        gradcam = np.zeros((84, 84), dtype=np.float32)
        gradcam_step = 0

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

            if total_steps % gradcam_frequency == 0:
                gradcam = compute_gradcam(
                    network=agent.online_network,
                    state=state,
                    action=action,
                    device=device,
                    layer_name=gradcam_layer
                )
                gradcam_step = total_steps

            video_frame = create_xray_frame(
                rgb_frame=rgb_frame,
                state=state,
                gradcam=gradcam,
                q_values=q_values,
                action=action,
                step=total_steps,
                episode_reward=episode_reward,
                gradcam_step=gradcam_step,
                gradcam_layer=gradcam_layer
            )

            video_frame = cv2.resize(
                video_frame,
                (1920, 1080),
                interpolation=cv2.INTER_CUBIC)
            
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
        "--gradcam-frequency",
        type=int,
        default=5
    )

    parser.add_argument(
        "--gradcam-layer",
        type=str,
        choices=["features.0", "features.2", "features.4"],
        default="features.0"
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
        gradcam_frequency=args.gradcam_frequency,
        gradcam_layer=args.gradcam_layer,
        max_steps=args.max_steps
    )
