
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.utils.pong_environment import PongEnvironment


def test_pong_environment_checkpoint() -> None:
    """
    Verify that saving and restoring the Pong environment
    reproduces the same observations, rewards, and termination flags.
    """
    env = PongEnvironment(num_frames=4)

    try:
        env.reset()

        for _ in range(30):
            env.step(1)

        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "environment.pth"

            torch.save(env.state_dict(), checkpoint_path)

            actions = [0, 1, 2, 1, 0, 2, 2, 1] * 5

            original_results = []

            for action in actions:
                state, reward, terminated, truncated, _ = env.step(action)

                original_results.append(
                    (
                        state.copy(),
                        reward,
                        terminated,
                        truncated
                    )
                )

                assert not (terminated or truncated), (
                    "Episode ended during the checkpoint test."
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            env.load_state_dict(checkpoint)

            for index, action in enumerate(actions):
                state, reward, terminated, truncated, _ = env.step(action)

                expected_state, expected_reward, expected_terminated, (
                    expected_truncated
                ) = original_results[index]

                assert np.array_equal(state, expected_state), (
                    f"State mismatch at step {index + 1}"
                )

                assert reward == expected_reward, (
                    f"Reward mismatch at step {index + 1}"
                )

                assert terminated == expected_terminated, (
                    f"Termination mismatch at step {index + 1}"
                )

                assert truncated == expected_truncated, (
                    f"Truncation mismatch at step {index + 1}"
                )

        print("Pong environment checkpoint test passed!")

    finally:
        env.close()


if __name__ == "__main__":
    test_pong_environment_checkpoint()
