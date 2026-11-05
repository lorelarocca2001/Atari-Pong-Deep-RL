
from collections import deque

import numpy as np


class FrameStack:
    """Maintain a stack of the most recent preprocessed Atari frames."""

    def __init__(self, num_frames: int = 4):
        """
        Initialize the frame stack.

        Args:
            num_frames: Number of consecutive frames to maintain.
        """
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """
        Initialize the stack by repeating the first frame.

        Args:
            frame: First preprocessed frame with shape (84, 84).

        Returns:
            Initial frame stack with shape (num_frames, 84, 84).
        """
        self.frames.clear()

        for _ in range(self.num_frames):
            self.frames.append(frame.copy())

        return self.get_state()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a new frame and discard the oldest one.

        Args:
            frame: New preprocessed frame with shape (84, 84).

        Returns:
            Updated frame stack with shape (num_frames, 84, 84).
        """
        self.frames.append(frame.copy())
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Return the current stack in channel-first format.

        Returns:
            Array with shape (num_frames, 84, 84) and dtype uint8.
        """
        return np.stack(self.frames, axis=0)
