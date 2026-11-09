

import random

import numpy as np
import torch

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 5,
    epsilon: float = 0.01,
    episode_seeds: list[int] | None = None
) -> dict:
    """
    Evaluate a DQN agent without updating its parameters.

    Args:
        agent: DQN agent to evaluate.
        num_episodes: Number of evaluation episodes.
        epsilon: Fixed exploration probability during evaluation.
        episode_seeds: Optional seeds for reproducible episodes.

    Returns:
        Dictionary containing evaluation metrics and episode rewards.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1.")

    if episode_seeds is not None and len(episode_seeds) != num_episodes:
        raise ValueError(
            "Number of episode seeds must match num_episodes."
        )

    env = PongEnvironment(num_frames=4)
    rewards = []

    was_training = agent.online_network.training
    agent.online_network.eval()

    try:
        for episode in range(num_episodes):
            seed = (
                episode_seeds[episode]
                if episode_seeds is not None
                else None
            )

            rng = random.Random(seed) if seed is not None else None

            state = env.reset(seed=seed)
            episode_reward = 0.0
            done = False

            while not done:
                if rng is None:
                    action = agent.select_action(state, epsilon)
                elif rng.random() < epsilon:
                    action = rng.randrange(agent.num_actions)
                else:
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            state,
                            device=agent.device
                        ).unsqueeze(0)

                        q_values = agent.online_network(state_tensor)
                        action = int(q_values.argmax(dim=1).item())

                state, reward, terminated, truncated, _ = env.step(
                    action
                )

                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

    finally:
        env.close()
        agent.online_network.train(was_training)

    rewards_array = np.asarray(rewards, dtype=np.float64)

    return {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "median_reward": float(np.median(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "win_rate": float(np.mean(rewards_array > 0) * 100),
        "episode_rewards": rewards
    }

