
import gymnasium as gym
import ale_py
import numpy as np
import torch

from src.utils.ball_detector import detect_ball_x
from src.utils.frame_stack import FrameStack
from src.utils.preprocessing import preprocess_frame
from src.utils.rally_detector import RallyDetector


class PongEnvironment:
    """Manage Atari Pong interactions and optional rally reward shaping."""

    def __init__(
        self,
        num_frames: int = 4,
        rally_bonus: float = 0.0
    ):
        """
        Initialize the Pong environment and frame stack.

        Args:
            num_frames: Number of consecutive frames in each state.
            rally_bonus: Additional reward for a detected paddle hit.
        """
        if not np.isfinite(rally_bonus) or rally_bonus < 0:
            raise ValueError("rally_bonus must be non-negative and finite.")

        gym.register_envs(ale_py)

        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.frame_stack = FrameStack(num_frames=num_frames)

        self.action_mapping = [0, 2, 3]
        self.num_actions = len(self.action_mapping)

        self.rally_bonus = rally_bonus
        self.rally_detector = RallyDetector()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset Pong, the rally detector, and the frame stack.

        Args:
            seed: Optional random seed for the environment.

        Returns:
            Initial state with shape (num_frames, 84, 84).
        """
        observation, _ = self.env.reset(seed=seed)
        self.rally_detector.reset()

        frame = preprocess_frame(observation)

        return self.frame_stack.reset(frame)

    def step(self, action: int) -> tuple:
        """
        Execute an action and optionally apply rally reward shaping.

        Args:
            action: Agent action (0: NOOP, 1: UP, 2: DOWN).

        Returns:
            Tuple containing next state, shaped reward, terminated,
            truncated, and environment information.
        """
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action: {action}")

        ale_action = self.action_mapping[action]

        observation, reward, terminated, truncated, info = self.env.step(
            ale_action
        )

        original_reward = float(reward)
        hit = False

        if self.rally_bonus > 0.0:
            ball_x = detect_ball_x(observation)
            hit = self.rally_detector.update(ball_x)

            if original_reward != 0.0 or terminated or truncated:
                hit = False
                self.rally_detector.reset()

        shaped_reward = (
            original_reward + self.rally_bonus * float(hit)
        )

        info = dict(info)
        info["original_reward"] = original_reward
        info["rally_hit"] = hit
        info["shaping_bonus"] = shaped_reward - original_reward

        frame = preprocess_frame(observation)
        next_state = self.frame_stack.step(frame)

        return next_state, shaped_reward, terminated, truncated, info

    def state_dict(self) -> dict:
        """
        Serialize the Atari emulator, frame stack, and rally detector.

        Returns:
            Dictionary containing the environment training state.
        """
        ale_state = self.env.unwrapped.ale.cloneSystemState()
        serialized_state = ale_state.serialize()

        frame_stack_state = self.frame_stack.state_dict()

        return {
            "ale_state": torch.tensor(
                list(serialized_state),
                dtype=torch.uint8
            ),
            "frame_stack": {
                "num_frames": frame_stack_state["num_frames"],
                "frames": [
                    torch.from_numpy(frame.copy())
                    for frame in frame_stack_state["frames"]
                ]
            },
            "rally_bonus": self.rally_bonus,
            "rally_detector": {
                "previous_x": self.rally_detector.previous_x,
                "previous_direction": self.rally_detector.previous_direction
            }
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore the Atari emulator, frame stack, and rally detector.

        Args:
            state_dict: Previously serialized environment state.
        """
        checkpoint_bonus = float(state_dict.get("rally_bonus", 0.0))

        if checkpoint_bonus != self.rally_bonus:
            raise ValueError(
                f"Rally bonus mismatch: checkpoint uses "
                f"{checkpoint_bonus}, environment uses {self.rally_bonus}."
            )

        serialized_state = bytes(
            state_dict["ale_state"].cpu().tolist()
        )

        ale_state = ale_py.ALEState(serialized_state)

        self.env.unwrapped.ale.restoreSystemState(ale_state)

        frame_stack_state = state_dict["frame_stack"]

        self.frame_stack.load_state_dict({
            "num_frames": frame_stack_state["num_frames"],
            "frames": [
                frame.cpu().numpy().copy()
                for frame in frame_stack_state["frames"]
            ]
        })

        detector_state = state_dict.get("rally_detector")

        if detector_state is None:
            self.rally_detector.reset()
        else:
            self.rally_detector.previous_x = detector_state["previous_x"]
            self.rally_detector.previous_direction = detector_state[
                "previous_direction"
            ]

    def close(self) -> None:
        """Release the Atari environment resources."""
        self.env.close()


