
import math

class LinearEpsilonScheduler:
    """Manage linear epsilon decay for DQN exploration."""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_steps: int = 1_000_000
    ):
        """
        Initialize the linear epsilon schedule.

        Args:
            epsilon_start: Initial exploration probability.
            epsilon_end: Minimum exploration probability.
            decay_steps: Number of steps required to complete the decay.
        """
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("Invalid epsilon range.")

        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive.")

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps

    def get_epsilon(self, step: int) -> float:
        """
        Compute epsilon at a given training step.

        Args:
            step: Current training step.

        Returns:
            Exploration probability for the given step.
        """
        if step < 0:
            raise ValueError("Step cannot be negative.")

        fraction = min(step / self.decay_steps, 1.0)

        return (
            self.epsilon_start
            + fraction * (self.epsilon_end - self.epsilon_start)
        )


class ExponentialEpsilonScheduler:
    """Manage normalized exponential epsilon decay for DQN exploration."""

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_steps: int = 1_000_000,
        decay_rate: float = 5.0
    ):
        """
        Initialize the exponential epsilon schedule.

        Args:
            epsilon_start: Initial exploration probability.
            epsilon_end: Final exploration probability.
            decay_steps: Number of steps required to complete the decay.
            decay_rate: Controls how rapidly epsilon decreases.
        """
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("Invalid epsilon range.")

        if decay_steps <= 0:
            raise ValueError("decay_steps must be positive.")

        if not math.isfinite(decay_rate) or decay_rate <= 0:
            raise ValueError("decay_rate must be positive and finite.")

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_epsilon(self, step: int) -> float:
        """
        Compute epsilon using normalized exponential decay.

        Args:
            step: Current training step.

        Returns:
            Exploration probability for the given step.
        """
        if step < 0:
            raise ValueError("Step cannot be negative.")

        fraction = min(step / self.decay_steps, 1.0)

        numerator = (
            math.exp(-self.decay_rate * fraction)
            - math.exp(-self.decay_rate)
        )

        denominator = 1.0 - math.exp(-self.decay_rate)

        return (
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end)
            * numerator / denominator
        )


class ConstantEpsilonScheduler:
    """Maintain a constant exploration probability."""

    def __init__(self, epsilon: float = 0.1):
        """
        Initialize the constant epsilon schedule.

        Args:
            epsilon: Fixed exploration probability.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.epsilon = epsilon

    def get_epsilon(self, step: int) -> float:
        """
        Return the constant exploration probability.

        Args:
            step: Current training step.

        Returns:
            Fixed exploration probability.
        """
        if step < 0:
            raise ValueError("Step cannot be negative.")

        return self.epsilon

