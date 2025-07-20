import videostreaming  # Ensure the environment is registered
import gymnasium as gym

env = gym.make("VideoStreaming-v0", trace_name="fcc", bandwidth_type="high", qoe_type="normal", seed=42)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, info)
