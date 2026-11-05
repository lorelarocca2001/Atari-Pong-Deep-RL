
import numpy as np

from src.utils.pong_environment import PongEnvironment


def test_pong_environment():
    """
    Verify Pong environment initialization, actions, and state transitions.

    Checks observation shapes, data types, rewards, and valid action handling.
    """
    env = PongEnvironment(num_frames=4)

    try:
        state = env.reset()

        assert state.shape == (4, 84, 84)
        assert state.dtype == np.uint8
        assert env.num_actions == 3

        print("Initial state shape:", state.shape)
        print("Number of actions:", env.num_actions)

        total_reward = 0.0

        for step in range(100):
            action = np.random.randint(env.num_actions)

            next_state, reward, terminated, truncated, info = env.step(action)

            assert next_state.shape == (4, 84, 84)
            assert next_state.dtype == np.uint8
            assert reward in (-1, 0, 1)

            total_reward += reward

            if terminated or truncated:
                next_state = env.reset()

            state = next_state

        print("Steps executed:", 100)
        print("Total reward:", total_reward)
        print("Pong environment test passed!")

    finally:
        env.close()


if __name__ == "__main__":
    test_pong_environment()
