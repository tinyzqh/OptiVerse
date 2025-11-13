from gymnasium.envs.registration import register

register(id="VideoStreaming-v0", entry_point="OptiVerse.envs.video.VideoStreaming:VideoStreamingEnv")
register(id="ILBanditTask-v0", entry_point="OptiVerse.envs.toys.BanditTask:BanditTaskEnv")
register(id="ILBanditRewardTask-v0", entry_point="OptiVerse.envs.toys.BanditTaskReward:BanditTaskRewardEnv")
register(id="FourRoom-v0", entry_point="OptiVerse.envs.grid.fourrooms:FourRoomsEnv")
register(id="NeRFLego-v0", entry_point="OptiVerse.envs.nerf.LegoTask:LegoTaskEnv")
