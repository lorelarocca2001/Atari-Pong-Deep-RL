
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
