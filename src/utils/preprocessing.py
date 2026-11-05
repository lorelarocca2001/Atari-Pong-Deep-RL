
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert an RGB Atari frame to an 84x84 grayscale image.

    Args:
        frame: Raw RGB observation with shape (210, 160, 3).

    Returns:
        Preprocessed grayscale frame with shape (84, 84) and dtype uint8.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    return resized.astype(np.uint8)
