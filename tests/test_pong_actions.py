
from pathlib import Path

import cv2
import numpy as np

from src.utils.pong_environment import PongEnvironment
from src.utils.preprocessing import preprocess_frame


def test_pong_actions():
    """
    Capture Atari Pong paddle movements for visual inspection.

    Saves before-and-after screenshots for the UP and DOWN
    actions to verify the agent's action mapping.
    """
    output_dir = Path("experiments/action_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PongEnvironment(num_frames=4)

    try:
        for action, name in [(1, "up"), (2, "down")]:
            env.reset()

            observation, _, _, _, _ = env.env.step(1)
            env.frame_stack.reset(preprocess_frame(observation))

            for _ in range(10):
                env.step(0)

            before = env.env.render().copy()

            for _ in range(30):
                _, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break

            after = env.env.render().copy()

            before = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
            after = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)

            comparison = np.concatenate(
                [before, after], axis=1
            )

            comparison = cv2.resize(
                comparison, (640, 420),
                interpolation=cv2.INTER_NEAREST
            )

            path = output_dir / f"pong_{name}.png"
            cv2.imwrite(str(path), comparison)

            print(f"Saved: {path}")

        print("Pong action test completed!")

    finally:
        env.close()


if __name__ == "__main__":
    test_pong_actions()
