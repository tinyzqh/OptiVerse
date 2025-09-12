import pytest
import gymnasium as gym


import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import OptiVerse


def test_videostreaming_env_runs():
    env = gym.make("VideoStreaming-v0", trace_name="fcc", bandwidth_type="high", qoe_type="normal", seed=42)
    obs, info = env.reset()
    assert obs is not None

    done = False
    steps = 0
    while not done and steps < 10:  # 跑几个 step 就好
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        steps += 1
        if terminated or truncated:
            done = True

    env.close()
