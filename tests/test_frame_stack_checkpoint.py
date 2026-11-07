
import numpy as np

from src.utils.frame_stack import FrameStack


def test_frame_stack_checkpoint() -> None:
    """Verify that frame stack serialization preserves stored observations."""
    original = FrameStack(num_frames=4)

    first_frame = np.full((84, 84), 10, dtype=np.uint8)
    original.reset(first_frame)

    for value in [20, 30, 40, 50]:
        frame = np.full((84, 84), value, dtype=np.uint8)
        original.step(frame)

    saved_state = original.state_dict()

    restored = FrameStack(num_frames=4)
    restored.load_state_dict(saved_state)

    assert restored.num_frames == original.num_frames
    assert np.array_equal(restored.get_state(), original.get_state())

    expected_values = [20, 30, 40, 50]

    for frame, value in zip(restored.get_state(), expected_values):
        assert np.all(frame == value)

    next_frame = np.full((84, 84), 60, dtype=np.uint8)

    original.step(next_frame)
    restored.step(next_frame)

    assert np.array_equal(restored.get_state(), original.get_state())

    print("Frame stack checkpoint test passed!")


if __name__ == "__main__":
    test_frame_stack_checkpoint()
