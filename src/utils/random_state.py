
import random

import numpy as np
import torch


def get_random_state() -> dict:
    """
    Capture Python, NumPy, and PyTorch random generator states.

    Returns:
        Dictionary containing the current RNG states.
    """
    numpy_state = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4]
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None
    }


def set_random_state(state: dict) -> None:
    """
    Restore Python, NumPy, and PyTorch random generator states.

    Args:
        state: Previously captured RNG states.
    """
    random.setstate(state["python"])

    numpy_state = state["numpy"]

    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy().copy(),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"]
    ))

    torch.set_rng_state(state["torch_cpu"].cpu())

    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG states, but CUDA is unavailable."
            )

        torch.cuda.set_rng_state_all([
            rng_state.cpu()
            for rng_state in state["torch_cuda"]
        ])
