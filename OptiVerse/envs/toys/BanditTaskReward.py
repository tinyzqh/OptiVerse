import random
import numpy as np
import gymnasium as gym


class BanditTaskRewardEnv(gym.Env):
    def __init__(self, num_points, seed):
        super(BanditTaskRewardEnv, self).__init__()
        self.seed(seed)

        self.num_points = num_points
        self.each_num = int(num_points / 4)

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def seed(self, seed_num):
        self.seed_num = seed_num
        if seed_num is not None:
            np.random.seed(seed_num)
            random.seed(seed_num)
            if hasattr(self, "action_space"):
                self.action_space.seed(seed_num)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        pos = 0.8
        std = 0.05

        left_up_samples = np.random.normal(loc=[-pos, pos], scale=[std, std], size=(self.each_num, 2))
        left_up_samples = np.clip(left_up_samples, -1.0, 1.0).astype(np.float32)

        left_bottom_samples = np.random.normal(loc=[-pos, -pos], scale=[std, std], size=(self.each_num, 2))
        left_bottom_samples = np.clip(left_bottom_samples, -1.0, 1.0).astype(np.float32)

        right_up_samples = np.random.normal(loc=[pos, pos], scale=[std, std], size=(self.each_num, 2))
        right_up_samples = np.clip(right_up_samples, -1.0, 1.0).astype(np.float32)

        right_bottom_samples = np.random.normal(loc=[pos, -pos], scale=[std, std], size=(self.each_num, 2))
        right_bottom_samples = np.clip(right_bottom_samples, -1.0, 1.0).astype(np.float32)

        data = np.concatenate([left_up_samples, left_bottom_samples, right_up_samples, right_bottom_samples], axis=0).astype(np.float32)

        self.action = data
        self.state = np.zeros_like(self.action, dtype=np.float32)

        r_left_up = 3.0 + 0.5 * np.random.randn(self.each_num, 1)
        r_left_bottom = 0.5 * np.random.randn(self.each_num, 1)
        r_right_up = 1.5 + 0.5 * np.random.randn(self.each_num, 1)
        r_right_bottom = 5.0 + 0.5 * np.random.randn(self.each_num, 1)
        self.reward = torch.cat([r_left_up, r_left_bottom, r_right_up, r_right_bottom], dim=0)
        return self.state[0], {}

    def step(self, batch_size):
        ind = np.random.randint(0, self.num_points, size=batch_size)
        return self.state[ind], self.action[ind], self.reward[ind], {}
