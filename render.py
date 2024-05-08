#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import subprocess
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, msssim
from lpipsPyTorch import lpips


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scale_mod, save=True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders", f'{scale_mod}')
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt", f'{scale_mod}')

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    l1_losses = []
    psnrs = []
    ssims = []
    lpipses = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view[1].cuda(), gaussians, pipeline, background, scaling_modifier=scale_mod)["render"]
        rendering = rendering.cpu()
        gt = view[0][0:3, :, :]
        l1_losses.append(l1_loss(rendering, gt))
        psnrs.append(psnr(rendering, gt).mean())
        ssims.append(ssim(rendering, gt))
        lpipses.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0)).mean())
        print(f'L1: {l1_losses[-1]}')
        print(f'PSNR: {psnrs[-1]}')
        print(f'SSIM: {ssims[-1]}')
        print(f'LPIPS: {lpipses[-1]}')
        if save:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".jpg"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".jpg"))
    print(f'mean_L1: {torch.stack(l1_losses).mean()}')
    print(f'mean_PSNR: {torch.stack(psnrs).mean()}')
    print(f'mean_SSIM: {torch.stack(ssims).mean()}')
    print(f'mean_LPIPS: {torch.stack(lpipses).mean()}')
    
    if save:
        cmd = ['ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(render_path, '%05d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(render_path, f'../render-{scale_mod}.mp4')]
        subprocess.run(cmd)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, time_duration: list, scale_mod: float, test_cameras: list):
    dataset.test_cams = test_cameras
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False, time_duration=time_duration)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scale_mod)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scale_mod)

        if False:
            from plyfile import PlyData, PlyElement
            import numpy as np
            ply_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), '4dgs.ply')
            os.makedirs(os.path.dirname(ply_path), exist_ok=True)

            l = ['x', 'y', 'z', 'mu_t', 'nx', 'ny', 'nz']
            for i in range(3):
                l.append('f_dc_{}'.format(i))
            l.append('opacity')
            for i in range(3):
                l.append('scale_{}'.format(i))
            l.append('t_scale')
            for i in range(4):
                l.append('rot_l_{}'.format(i))
            for i in range(4):
                l.append('rot_r_{}'.format(i))
            dtype = [(attr, 'f4') for attr in l]

            xyz = gaussians._xyz.detach().cpu().numpy()
            mu_t = gaussians._t.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = gaussians._features_dc.detach().cpu().numpy().reshape(-1, 3)
            opacity = gaussians._opacity.detach().cpu().numpy()
            scale = gaussians._scaling.detach().cpu().numpy()
            t_scale = gaussians._scaling_t.detach().cpu().numpy()
            rot_l = gaussians._rotation.detach().cpu().numpy()
            rot_r = gaussians._rotation_r.detach().cpu().numpy()

            elements = np.empty(xyz.shape[0], dtype=dtype)
            attr = np.concatenate([xyz, mu_t, normals, f_dc, opacity, scale, t_scale, rot_l, rot_r], axis=1).astype('float32')
            elements[:] = list(map(tuple, attr))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(ply_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--time_duration", default=[0, 1], nargs=2, type=int, help="Time duration for the rendering")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scale_mod", type=float, default=1.0)
    parser.add_argument("--test_cameras", type=str, nargs='+', default=['00'], help="List of test cameras")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.time_duration, args.scale_mod, args.test_cameras)