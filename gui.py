from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import to_cuda
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
from easyvolcap.runners.custom_viewer import Viewer

import os
import argparse
from omegaconf import OmegaConf
from typing import NamedTuple
# from arguments import ModelParams, PipelineParams, get_combined_args

import json
import torch
import numpy as np
from scene.dataset_readers import readCamerasFromEasyVolcap
from scene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import GaussianModel, render

# a = '{"H":2032,"W":3840,"K":[[4279.6650390625,0.0,1920.0],[0.0,4279.6650390625,992.4420776367188],[0.0,0.0,1.0]],"R":[[0.41155678033828735,0.911384105682373,0.0],[-0.8666263818740845,0.39134538173675537,0.3095237910747528],[0.2820950746536255,-0.12738661468029022,0.9508903622627258]],"T":[[-4.033830642700195],[-1.7978200912475586],[3.9347341060638428]],"n":0.10000000149011612,"f":1000.0,"t":0.0,"v":0.0,"bounds":[[-10.0,-10.0,-3.0],[10.0,10.0,4.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":1.0,"movement_torque":1.0,"movement_speed":5.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,0.0,-1.0]}'),

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


class CustomViewer(Viewer):
    def __init__(self,
                 window_size=[1080, 1920],  # height, width
                 window_title: str = f'EasyVolcap Viewer Custom Window',  # MARK: global config
                 fullscreen: bool = False,
                 camera_cfg: dotdict = None,
                 pipeline: PipelineParams = None,
                 gaussian: GaussianModel = None,
                 duration: list = [0., 10.],
                ):
        super(CustomViewer, self).__init__(
            window_size=window_size,
            window_title=window_title,
            fullscreen=fullscreen,
            camera_cfg=camera_cfg)
        self.pipeline = pipeline
        self.pipeline.env_map_res = 0
        self.gaussian = gaussian
        self.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        self.duration = duration

    def custom_render(self, batch):
        """
        def convert_to_gaussian_camera(K: torch.Tensor,
                               R: torch.Tensor,
                               T: torch.Tensor,
                               H: torch.Tensor,
                               W: torch.Tensor,
                               n: torch.Tensor,
                               f: torch.Tensor,
                               cpu_K: torch.Tensor,
                               cpu_R: torch.Tensor,
                               cpu_T: torch.Tensor,
                               cpu_H: int,
                               cpu_W: int,
                               cpu_n: float = 0.01,
                               cpu_f: float = 100.,
                               ):

        gaussian_camera = convert_to_gaussian_camera(
            K=to_cuda(batch.K),
            R=to_cuda(batch.R),
            T=to_cuda(batch.T),
            H=to_cuda(batch.H),
            W=to_cuda(batch.W),
            n=torch.tensor(0.01, device='cuda', dtype=torch.float32),
            f=torch.tensor(100., device='cuda', dtype=torch.float32),
            cpu_K=batch.K,
            cpu_R=batch.R,
            cpu_T=batch.T,
            cpu_H=batch.H,
            cpu_W=batch.W,
            cpu_n=0.01,
            cpu_f=100.,
        )
        """
        K_np = batch.K.numpy()
        R_np = batch.R.numpy()
        T_np = batch.T.numpy().reshape(3,)
        gaussian_camera = Camera(
            colmap_id=0,
            R=R_np,
            T=T_np,
            FoVx=-1.0,
            FoVy=-1.0,
            image=None,
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
            data_device="cuda",
            timestamp=batch.meta.t.item() * (self.duration[1] - self.duration[0]) + self.duration[0],
            cx=K_np[0, 2],
            cy=K_np[1, 2],
            fl_x=K_np[0, 0],
            fl_y=K_np[1, 1],
            depth=None,
            resolution=[batch.W.item(), batch.H.item()],
            image_path=None,
            meta_only=True,
        )
        out = render(gaussian_camera.cuda(), self.gaussian, self.pipeline, self.background)["render"]
        out = out.permute(1, 2, 0).contiguous()
        out = torch.cat([out, torch.ones_like(out[..., :1])], dim=-1)
        return out


@catch_throw
def main(cfg):
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

    # with open(os.path.join(model.source_path, "transforms_test.json")) as json_file:
    #     contents = json.load(json_file)
    # frames = contents['frames']
    # idx = 0
    # frame = frames[0]
    # timestamp = frame.get('time', 0.0)
    # if model.frame_ratio > 1:
    #     timestamp /= model.frame_ratio
    # w2c = np.array(frame["transform_matrix"])
    # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    # T = w2c[:3, 3]
    # FovX = FovY = -1.0
    # fl_x = frame['fl_x']
    # fl_y = frame['fl_y']
    # cx = frame['cx']
    # cy = frame['cy']
    # test_cam_infos = [
    #     CameraInfo(
    #         uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
    #         image=None, depth=None,
    #         image_path="", 
    #         image_name="", 
    #         width=int(cx)*2, height=int(cy)*2, 
    #         timestamp=timestamp,
    #         fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy
    #     )
    # ]
    # test_cam = cameraList_from_camInfos(test_cam_infos, resolution_scale=1.0, args=model)

    cameras = read_camera(model.source_path)
    camera_names = sorted(list(cameras.keys()))
    default_camera = cameras[camera_names[0]]
    # downsample
    K = default_camera['K'].copy()
    K[:2] *= 0.5
    default_camera['K'] = K
    default_camera['W'] = int(K[0,2]*2)
    default_camera['H'] = int(K[1,2]*2)

    camera_cfg = dotdict(
        H=default_camera['H'],
        W=default_camera['W'],
        K=default_camera['K'],
        R=default_camera['R'],
        T=default_camera['T'],
    )

    viewer = CustomViewer(
        window_size=[default_camera['H'], default_camera['W']],
        camera_cfg=camera_cfg,
        pipeline=pipeline,
        gaussian=gaussians,
        duration=cfg.time_duration
    )
        
    viewer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom Viewer')
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
