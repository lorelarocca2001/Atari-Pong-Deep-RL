
import numpy as np

from src.training.epsilon_scheduler import LinearEpsilonScheduler


def test_epsilon_scheduler():
    """
    Verify linear epsilon decay and parameter validation.

    Checks initial, intermediate, and final epsilon values,
    monotonic decay, and invalid input handling.
    """
    scheduler = LinearEpsilonScheduler(
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_steps=1_000_000
    )

    test_steps = [0, 250_000, 500_000, 750_000, 1_000_000, 1_500_000]
    expected_values = [1.0, 0.775, 0.55, 0.325, 0.1, 0.1]

    for step, expected in zip(test_steps, expected_values):
        epsilon = scheduler.get_epsilon(step)

        assert np.isclose(epsilon, expected)
        print(f"Step {step:>9}: epsilon = {epsilon:.4f}")

    epsilon_values = [
        scheduler.get_epsilon(step)
        for step in range(0, 1_500_001, 10_000)
    ]

    assert all(
        current >= following
        for current, following in zip(epsilon_values, epsilon_values[1:])
    )

    assert all(0.1 <= epsilon <= 1.0 for epsilon in epsilon_values)

    for invalid_step in [-1, -100]:
        try:
            scheduler.get_epsilon(invalid_step)
        except ValueError:
            pass
        else:
            raise AssertionError("Negative step was accepted")

    invalid_configs = [
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"epsilon_start": 1.2, "epsilon_end": 0.1},
        {"epsilon_start": 1.0, "epsilon_end": -0.1},
        {"decay_steps": 0}
    ]

    for config in invalid_configs:
        try:
            LinearEpsilonScheduler(**config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid configuration accepted: {config}")

    print("Monotonic decay verified!")
    print("Epsilon range verified!")
    print("Invalid parameters rejected!")
    print("Epsilon scheduler test passed!")


if __name__ == "__main__":
    test_epsilon_scheduler()
