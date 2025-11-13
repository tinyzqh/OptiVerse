import sys, os
import gymnasium as gym


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import OptiVerse


def test_nerf_lego_task_env_runs():
    env = gym.make("NeRFLego-v0", seed=42)
    obs, info = env.reset(seed=42)

    env.close()


if __name__ == "__main__":
    test_nerf_lego_task_env_runs()
