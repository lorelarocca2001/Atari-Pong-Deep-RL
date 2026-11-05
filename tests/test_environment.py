import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

state, info = env.reset()

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Action meanings:", env.unwrapped.get_action_meanings())
print("State shape:", state.shape)

for step in range(100):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        next_state, info = env.reset()

env.close()

print("Test completato!")
