
import gymnasium as gym
import ale_py
import numpy as np

from src.utils.preprocessing import preprocess_frame
from src.utils.frame_stack import FrameStack


def test_frame_stack():
    """
    Verify frame stacking initialization and updates using Atari Pong.

    Checks the output shape, data type, initial frame repetition,
    and FIFO behavior when a new frame is added.
    """
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

    try:
        observation, _ = env.reset()

        frame_stack = FrameStack(num_frames=4)
        frame = preprocess_frame(observation)
        state = frame_stack.reset(frame)

        print("Initial stack shape:", state.shape)
        print("Initial stack dtype:", state.dtype)

        assert state.shape == (4, 84, 84)
        assert state.dtype == np.uint8
        assert all(np.array_equal(state[0], state[i]) for i in range(1, 4))

        action = env.action_space.sample()
        next_observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            next_observation, _ = env.reset()
            next_state = frame_stack.reset(preprocess_frame(next_observation))
        else:
            next_frame = preprocess_frame(next_observation)
            next_state = frame_stack.step(next_frame)

            assert np.array_equal(next_state[:3], state[1:])
            assert np.array_equal(next_state[3], next_frame)

        print("Updated stack shape:", next_state.shape)
        print("Frame stacking test passed!")

    finally:
        env.close()


if __name__ == "__main__":
    test_frame_stack()
