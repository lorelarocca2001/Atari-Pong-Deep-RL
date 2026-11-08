
import random
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.datasets.replay_buffer import ReplayBuffer
from src.training.dqn_agent import DQNAgent
from src.training.train import load_checkpoint, save_checkpoint
from src.utils.pong_environment import PongEnvironment


def test_training_checkpoint() -> None:
    """
    Verify that a complete DQN checkpoint restores training progress,
    networks, optimizer, replay buffer, environment, and random states.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    original_env = PongEnvironment(num_frames=4)
    restored_env = PongEnvironment(num_frames=4)

    try:
        original_state = original_env.reset()

        original_agent = DQNAgent(
            num_actions=original_env.num_actions,
            device=device
        )

        original_buffer = ReplayBuffer(capacity=50)

        for i in range(30):
            action = i % 3

            next_state, reward, terminated, truncated, _ = (
                original_env.step(action)
            )

            done = terminated or truncated

            original_buffer.push(
                original_state,
                action,
                reward,
                next_state,
                done
            )

            original_state = (
                original_env.reset() if done else next_state
            )

        batch = original_buffer.sample(8, device)
        last_loss = original_agent.train_step(batch)

        episode_rewards = deque([-21.0, -19.0], maxlen=100)
        episode_reward = -2.0
        episode_count = 2
        step = 1000

        with TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)

            save_checkpoint(
                agent=original_agent,
                buffer=original_buffer,
                env=original_env,
                step=step,
                directory=checkpoint_dir,
                state=original_state,
                episode_reward=episode_reward,
                episode_count=episode_count,
                episode_rewards=episode_rewards,
                last_loss=last_loss
            )

            checkpoint_path = checkpoint_dir / "dqn_step_1000.pth"

            assert checkpoint_path.is_file()

            python_value = random.random()
            numpy_value = np.random.random()
            torch_value = torch.rand(1)

            actions = [0, 1, 2, 1, 0, 2, 1, 0]
            expected_results = []

            for action in actions:
                result = original_env.step(action)

                expected_results.append((
                    result[0].copy(),
                    result[1],
                    result[2],
                    result[3]
                ))

                assert not (result[2] or result[3])

            restored_env.reset()

            restored_agent = DQNAgent(
                num_actions=restored_env.num_actions,
                device=device
            )

            restored_buffer = ReplayBuffer(capacity=10)

            restored = load_checkpoint(
                agent=restored_agent,
                buffer=restored_buffer,
                env=restored_env,
                checkpoint_path=str(checkpoint_path),
                device=device
            )

            assert restored["step"] == step
            assert restored["episode_count"] == episode_count
            assert restored["episode_reward"] == episode_reward
            assert list(restored["episode_rewards"]) == list(
                episode_rewards
            )
            assert restored["last_loss"] == last_loss
            assert np.array_equal(
                restored["state"],
                original_state
            )

            assert len(restored_buffer) == len(original_buffer)
            assert restored_buffer.buffer.maxlen == 50

            for expected, actual in zip(
                original_buffer.buffer,
                restored_buffer.buffer
            ):
                assert np.array_equal(expected.state, actual.state)
                assert expected.action == actual.action
                assert expected.reward == actual.reward
                assert np.array_equal(
                    expected.next_state,
                    actual.next_state
                )
                assert expected.done == actual.done

            for expected, actual in zip(
                original_agent.online_network.parameters(),
                restored_agent.online_network.parameters()
            ):
                assert torch.equal(expected, actual)

            for expected, actual in zip(
                original_agent.target_network.parameters(),
                restored_agent.target_network.parameters()
            ):
                assert torch.equal(expected, actual)

            original_optimizer = original_agent.optimizer.state_dict()
            restored_optimizer = restored_agent.optimizer.state_dict()

            assert (
                original_optimizer["param_groups"]
                == restored_optimizer["param_groups"]
            )

            for key, original_values in original_optimizer["state"].items():
                restored_values = restored_optimizer["state"][key]

                for name, expected in original_values.items():
                    actual = restored_values[name]

                    if isinstance(expected, torch.Tensor):
                        assert torch.equal(expected.cpu(), actual.cpu())
                    else:
                        assert expected == actual

            assert random.random() == python_value
            assert np.random.random() == numpy_value
            assert torch.equal(torch.rand(1), torch_value)

            for index, action in enumerate(actions):
                state, reward, terminated, truncated, _ = (
                    restored_env.step(action)
                )

                expected_state, expected_reward, (
                    expected_terminated
                ), expected_truncated = expected_results[index]

                assert np.array_equal(state, expected_state), (
                    f"Environment state mismatch at action {index + 1}"
                )

                assert reward == expected_reward
                assert terminated == expected_terminated
                assert truncated == expected_truncated

        print("Complete training checkpoint test passed!")

    finally:
        original_env.close()
        restored_env.close()


if __name__ == "__main__":
    test_training_checkpoint()
