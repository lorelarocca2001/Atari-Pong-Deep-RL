
import numpy as np
import torch

from src.training.dqn_agent import DQNAgent
from src.utils.pong_environment import PongEnvironment


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 5,
    epsilon: float = 0.01
) -> dict:
    """
    Evaluate a DQN agent without updating its parameters.

    Args:
        agent: DQN agent to evaluate.
        num_episodes: Number of evaluation episodes.
        epsilon: Fixed exploration probability during evaluation.

    Returns:
        Dictionary containing mean reward, standard deviation,
        and individual episode rewards.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("Epsilon must be between 0 and 1.")

    env = PongEnvironment(num_frames=4)
    rewards = []

    was_training = agent.online_network.training
    agent.online_network.eval()

    try:
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                with torch.no_grad():
                    action = agent.select_action(state, epsilon)

                state, reward, terminated, truncated, _ = env.step(action)

                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

    finally:
        env.close()
        agent.online_network.train(was_training)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "episode_rewards": rewards
    }
