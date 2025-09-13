import pytest
import sys, os
import gymnasium as gym


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import OptiVerse


def test_bandit_task_env_runs():
    env = gym.make("ILBanditTask-v0", num_points=10000, seed=42)
    env.reset(seed=42)
    state, action, reward, _ = env.step(100)
    print(state[0])

    env.close()
