
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.utils.random_state import get_random_state, set_random_state


def test_random_state() -> None:
    """
    Verify that random generator states can be saved and restored
    while preserving Python, NumPy, and PyTorch random sequences.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    saved_state = get_random_state()

    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "random_state.pth"

        torch.save(saved_state, checkpoint_path)

        python_values = [random.random() for _ in range(10)]
        numpy_values = np.random.random(10)
        torch_values = torch.rand(10)

        if torch.cuda.is_available():
            cuda_values = torch.rand(10, device="cuda:0")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        set_random_state(checkpoint)

        restored_python_values = [random.random() for _ in range(10)]
        restored_numpy_values = np.random.random(10)
        restored_torch_values = torch.rand(10)

        assert python_values == restored_python_values
        assert np.array_equal(numpy_values, restored_numpy_values)
        assert torch.equal(torch_values, restored_torch_values)

        if torch.cuda.is_available():
            restored_cuda_values = torch.rand(10, device="cuda:0")
            assert torch.equal(cuda_values, restored_cuda_values)

    print("Random state checkpoint test passed!")


if __name__ == "__main__":
    test_random_state()
