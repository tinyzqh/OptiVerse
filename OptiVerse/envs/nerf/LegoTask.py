import os
import cv2
import json
import random
import imageio
import numpy as np
import gymnasium as gym
from pathlib import Path

trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=np.float32)

rot_phi = lambda phi: np.array([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]], dtype=np.float32)

rot_theta = lambda th: np.array([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]], dtype=np.float32)


class LegoTaskEnv(gym.Env):
    def __init__(self, seed=None):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.daradir = os.path.abspath(os.path.join(self.project_root, "datasets/nerf/lego"))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.testskip = 8
        self.half_res = True

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

        splits = ["train", "val", "test"]
        metas = {}
        for s in splits:
            with open(os.path.join(self.daradir, "transforms_{}.json".format(s)), "r") as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]

        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s == "train" or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            for frame in meta["frames"][::skip]:
                fname = os.path.join(self.daradir, frame["file_path"] + ".png")
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame["transform_matrix"]))
            imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        render_poses = np.stack([self._pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], axis=0)

        if self.half_res:
            H = H // 2
            W = W // 2
            focal = focal / 2.0

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        return np.array([0]), {"img": imgs, "poses": poses, "render_poses": render_poses, "hwf": [H, W, focal], "i_split": i_split}

    def _pose_spherical(self, theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
        c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
        swap = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        c2w = swap @ c2w
        # c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w
