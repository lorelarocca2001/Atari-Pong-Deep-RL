
import gymnasium as gym
import ale_py
import numpy as np
import torch

from src.utils.preprocessing import preprocess_frame
from src.utils.frame_stack import FrameStack


class PongEnvironment:
    """Manage Atari Pong interactions and preprocess observations for DQN."""

    def __init__(self, num_frames: int = 4):
        """
        Initialize the Pong environment and frame stack.

        Args:
            num_frames: Number of consecutive frames in each state.
        """
        gym.register_envs(ale_py)

        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.frame_stack = FrameStack(num_frames=num_frames)

        self.action_mapping = [0, 2, 3]
        self.num_actions = len(self.action_mapping)

    def reset(self) -> np.ndarray:
        """
        Reset Pong and initialize the frame stack.

        Returns:
            Initial state with shape (num_frames, 84, 84).
        """
        observation, _ = self.env.reset()
        frame = preprocess_frame(observation)

        return self.frame_stack.reset(frame)

    def step(self, action: int) -> tuple:
        """
        Execute an action and update the frame stack.

        Args:
            action: Agent action (0: NOOP, 1: UP, 2: DOWN).

        Returns:
            Tuple containing next state, reward, terminated,
            truncated, and environment information.
        """
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action: {action}")

        ale_action = self.action_mapping[action]

        observation, reward, terminated, truncated, info = self.env.step(
            ale_action
        )

        frame = preprocess_frame(observation)
        next_state = self.frame_stack.step(frame)

        return next_state, reward, terminated, truncated, info

    def state_dict(self) -> dict:
        """
        Serialize the Atari emulator and frame stack states.

        Returns:
            Dictionary containing the serialized ALE state and frame stack.
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
            }
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore the Atari emulator and frame stack states.

        Args:
            state_dict: Previously serialized environment state.
        """
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

    def close(self) -> None:
        """Release the Atari environment resources."""
        self.env.close()

