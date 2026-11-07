
import numpy as np
import torch

from src.datasets.replay_buffer import ReplayBuffer


def test_replay_buffer_checkpoint() -> None:
    """Verify that replay buffer serialization preserves all transitions."""
    original = ReplayBuffer(capacity=100)

    for i in range(50):
        state = np.full((4, 84, 84), i, dtype=np.uint8)
        next_state = np.full((4, 84, 84), i + 1, dtype=np.uint8)

        original.push(
            state=state,
            action=i % 3,
            reward=float(i % 3 - 1),
            next_state=next_state,
            done=(i % 10 == 0)
        )

    saved_state = original.state_dict()

    restored = ReplayBuffer(capacity=10)
    restored.load_state_dict(saved_state)

    assert len(restored) == len(original)
    assert restored.buffer.maxlen == original.buffer.maxlen

    for expected, actual in zip(original.buffer, restored.buffer):
        assert np.array_equal(expected.state, actual.state)
        assert expected.action == actual.action
        assert expected.reward == actual.reward
        assert np.array_equal(expected.next_state, actual.next_state)
        assert expected.done == actual.done

    print("Replay buffer checkpoint test passed!")


if __name__ == "__main__":
    test_replay_buffer_checkpoint()
