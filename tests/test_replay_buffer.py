
import numpy as np
import torch

from src.datasets.replay_buffer import ReplayBuffer


def test_replay_buffer():
    """
    Verify replay buffer storage, capacity, and random sampling.

    Checks minibatch shapes, data types, device placement,
    and transition consistency.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    capacity = 100
    batch_size = 32

    buffer = ReplayBuffer(capacity=capacity)

    for i in range(150):
        state = np.full((4, 84, 84), i % 256, dtype=np.uint8)
        next_state = np.full((4, 84, 84), (i + 1) % 256, dtype=np.uint8)

        action = i % 3
        reward = float(i % 3 - 1)
        done = i % 10 == 0

        buffer.push(state, action, reward, next_state, done)

    assert len(buffer) == capacity

    states, actions, rewards, next_states, dones = buffer.sample(
        batch_size, device
    )

    assert states.shape == (batch_size, 4, 84, 84)
    assert next_states.shape == (batch_size, 4, 84, 84)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)

    assert states.dtype == torch.uint8
    assert next_states.dtype == torch.uint8
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32

    assert all(t.device == device for t in
               (states, actions, rewards, next_states, dones))

    assert torch.all((actions >= 0) & (actions < 3))
    assert torch.all((rewards >= -1) & (rewards <= 1))
    assert torch.all((dones == 0) | (dones == 1))

    assert torch.all(next_states[:, 0, 0, 0] == states[:, 0, 0, 0] + 1)

    sampled_ids = states[:, 0, 0, 0].cpu().numpy()
    assert len(np.unique(sampled_ids)) == batch_size
    assert np.all((sampled_ids >= 50) & (sampled_ids < 150))

    print("Device:", device)
    print("Buffer capacity:", capacity)
    print("Stored transitions:", len(buffer))
    print("Batch size:", batch_size)
    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    print("Rewards shape:", rewards.shape)
    print("Next states shape:", next_states.shape)
    print("Dones shape:", dones.shape)
    print("Replay buffer test passed!")


if __name__ == "__main__":
    test_replay_buffer()
