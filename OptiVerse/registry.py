from gymnasium.envs.registration import register

register(id="VideoStreaming-v0", entry_point="OptiVerse.envs.video.VideoStreaming:VideoStreamingEnv")
register(id="ILBanditTask-v0", entry_point="OptiVerse.envs.toys.BanditTask:BanditTaskEnv")
register(id="ILBanditRewardTask-v0", entry_point="OptiVerse.envs.toys.BanditTaskReward:BanditTaskRewardEnv")
