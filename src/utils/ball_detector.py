
import numpy as np


def detect_ball_x(frame: np.ndarray) -> float | None:
    """
    Detect the ball's horizontal position in a Pong RGB frame.

    Args:
        frame: Original RGB observation with shape (210, 160, 3).

    Returns:
        Ball center x-coordinate or None if detection is uncertain.
    """
    if frame.shape != (210, 160, 3):
        raise ValueError("Expected RGB frame with shape (210, 160, 3).")

    mask = np.all(frame >= 180, axis=2)
    mask[:34, :] = False
    mask[194:, :] = False
    mask[:, :18] = False
    mask[:, 140:] = False
    mask[:, 78:83] = False

    visited = np.zeros(mask.shape, dtype=bool)
    candidates = []

    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue

        stack = [(int(y), int(x))]
        visited[y, x] = True
        component = []

        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = cy + dy
                nx = cx + dx

                if (
                    0 <= ny < mask.shape[0]
                    and 0 <= nx < mask.shape[1]
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if not 2 <= len(component) <= 20:
            continue

        coordinates = np.asarray(component)
        height = np.ptp(coordinates[:, 0]) + 1
        width = np.ptp(coordinates[:, 1]) + 1

        if height <= 6 and width <= 6:
            candidates.append(float(np.mean(coordinates[:, 1])))

    if len(candidates) != 1:
        return None

    return candidates[0]
