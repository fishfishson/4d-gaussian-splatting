import os
from os.path import join
import argparse
from omegaconf import OmegaConf
from typing import NamedTuple
# from arguments import ModelParams, PipelineParams, get_combined_args

import json
import imageio
import torch
import numpy as np
from tqdm import tqdm
from scene.dataset_readers import readCamerasFromEasyVolcap
from scene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import GaussianModel, render
import torchvision

from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import generate_video


class ModelParams():
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.extension = ".png"
        self.ply_path = 'null.ply'
        self.num_extra_pts = 0
        self.loaded_pth = ""
        self.frame_ratio = 1
        self.dataloader = False
    
    def __repr__(self) -> str:
        print(f"Model Parameters:\n")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        return ""


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.env_map_res = 0
        self.env_optimize_until = 1000000000
        self.env_optimize_from = 0
        self.eval_shfs_4d = False
    
    def __repr__(self) -> str:
        print(f"Pipeline Parameters:\n")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        return ""


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0


def main(args, cfg):
    model = ModelParams()
    pipeline = PipelineParams()
    for k, v in cfg.ModelParams.items():
        if hasattr(model, k):
            setattr(model, k, v)
    for k, v in cfg.PipelineParams.items():
        if hasattr(pipeline, k):
            setattr(pipeline, k, v)

    sh_degree = model.sh_degree
    sh_degree_t = 2 if pipeline.eval_shfs_4d else 0
    gaussian_dim = cfg.gaussian_dim
    rot_4d = cfg.rot_4d
    gaussians = GaussianModel(sh_degree=sh_degree,
                              sh_degree_t=sh_degree_t,
                              gaussian_dim=gaussian_dim,
                              rot_4d=rot_4d)

    loaded_path = join(cfg.ModelParams.model_path, "chkpnt30000.pth")
    if os.path.exists(loaded_path):
        print(f'Load ckpt from {loaded_path}!')
        model_args = torch.load(loaded_path)[0]
        gaussians.restore(model_args, None)

    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
    cameras = read_camera(args.path)
    camera_names = sorted(list(cameras.keys()))

    seq = os.path.basename(args.config).split('.')[0]
    output_path = join(args.output, seq)
    os.makedirs(output_path, exist_ok=True)

    for k in tqdm(camera_names):
        camera = cameras[k]
        K_np = camera['K']
        R_np = camera['R']
        T_np = camera['T'].reshape(3,)
        H = camera['H']
        W = camera['W']
        gaussian_camera = Camera(
            colmap_id=0,
            R=R_np.T,
            T=T_np,
            FoVx=-1.0,
            FoVy=-1.0,
            image=None,
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
            data_device="cuda",
            timestamp=camera['t'] * (cfg.time_duration[1] - cfg.time_duration[0]) + cfg.time_duration[0],
            cx=K_np[0, 2],
            cy=K_np[1, 2],
            fl_x=K_np[0, 0],
            fl_y=K_np[1, 1],
            depth=None,
            resolution=[W, H],
            image_path=None,
            meta_only=True,
        )
        out = render(gaussian_camera.cuda(), gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(out, join(output_path, k + ".jpg"))
    result_str = f'"{output_path}/*jpg"'
    output_path = join(args.output, f'{seq}.mp4')
    try:
        generate_video(result_str, output_path, fps=60)  # one video for one type?
    except RuntimeError as e:
        print('Error encountered during video composition, will retry without hardware encoding')
        generate_video(result_str, output_path, fps=60, hwaccel='none', vcodec='libx265')  # one video for one type?

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render Novel')
    parser.add_argument("--config", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--output", type=str, default='novel')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(args, cfg)