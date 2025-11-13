# import pytest
import sys, os
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from OptiVerse.envs.video.VideoStreaming import VideoStreamingEnv


def test_videostreaming_env_local():
    env = VideoStreamingEnv(trace_name="train", bandwidth_type="hybrid", qoe_type="normal", seed=0)
    obs, info = env.reset()
    assert obs is not None

    done = False
    steps = 0
    while not done and steps < 10:
        action = 5
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        steps += 1
        if terminated or truncated:
            done = True

    env.close()


if __name__ == "__main__":
    test_videostreaming_env_local()
