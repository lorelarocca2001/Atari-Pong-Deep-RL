
import gymnasium as gym
import ale_py
import numpy as np

from src.utils.preprocessing import preprocess_frame


def test_preprocessing():
    """Verify the shape, dtype, and pixel range of a preprocessed Atari frame."""
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    try:
        state, _ = env.reset()
        processed = preprocess_frame(state)

        print("Original shape:", state.shape)
        print("Processed shape:", processed.shape)
        print("Processed dtype:", processed.dtype)
        print("Pixel range:", processed.min(), processed.max())

        assert processed.shape == (84, 84)
        assert processed.dtype == np.uint8
        assert processed.min() >= 0
        assert processed.max() <= 255

        print("Preprocessing test passed!")
    finally:
        env.close()


if __name__ == "__main__":
    test_preprocessing()
