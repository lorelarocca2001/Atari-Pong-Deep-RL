
class RallyDetector:
    """Detect potential ball bounces near the player's paddle."""

    def __init__(self, paddle_x_threshold: float = 115.0):
        """
        Initialize the rally detector.

        Args:
            paddle_x_threshold: Minimum ball x-coordinate near the paddle.
        """
        self.paddle_x_threshold = paddle_x_threshold
        self.previous_x = None
        self.previous_direction = 0

    def reset(self) -> None:
        """Reset the ball tracking state."""
        self.previous_x = None
        self.previous_direction = 0

    def update(self, ball_x: float | None) -> bool:
        """
        Detect a possible paddle hit from ball movement.

        Args:
            ball_x: Horizontal ball position in the original game frame.

        Returns:
            Whether a possible player paddle hit was detected.
        """
        if ball_x is None:
            self.reset()
            return False

        if self.previous_x is None:
            self.previous_x = ball_x
            return False

        delta_x = ball_x - self.previous_x

        if delta_x > 0:
            direction = 1
        elif delta_x < 0:
            direction = -1
        else:
            direction = 0

        hit = (
            self.previous_direction == 1
            and direction == -1
            and self.previous_x >= self.paddle_x_threshold
        )

        self.previous_x = ball_x

        if direction != 0:
            self.previous_direction = direction

        return hit
