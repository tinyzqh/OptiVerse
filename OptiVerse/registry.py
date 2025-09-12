import gym
from gym.envs.registration import register

register(id="VideoStreaming-v0", entry_point="OptiVerse.envs.video.VideoStreaming:VideoStreaming")