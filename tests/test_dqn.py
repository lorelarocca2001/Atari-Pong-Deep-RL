
import torch

from src.models.dqn import DQN


def test_dqn():
    """
    Verify DQN output dimensions and GPU compatibility.

    Checks Q-value shapes, numerical validity, and device placement.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DQN(num_actions=3).to(device)
    model.eval()

    batch = torch.randint(
        0, 256, (8, 4, 84, 84), dtype=torch.uint8, device=device
    )

    with torch.no_grad():
        q_values = model(batch)

    assert q_values.shape == (8, 3)
    assert torch.isfinite(q_values).all()

    print("Expected device:", device)
    print("Model device:", next(model.parameters()).device)
    print("Q-values device:", q_values.device)

    assert q_values.device == device

    print("Device:", device)
    print("Input shape:", batch.shape)
    print("Output shape:", q_values.shape)
    print("Q-values (first sample):", q_values[0])
    print("DQN test passed!")


if __name__ == "__main__":
    test_dqn()
