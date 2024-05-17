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
import sys
import json
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from train import eval_psnr, eval_lpips, eval_ssim

def render_set(output_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(output_path, name, "pred")
    gts_path = os.path.join(output_path, name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    psnrs = []
    lpips = []
    ssims = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view[1].cuda(), gaussians, pipeline, background)["render"]
        gt = view[0][0:3, :, :]
        image_path = view[1].image_path
        cam = os.path.dirname(image_path).split('/')[-1]
        image_name = os.path.basename(image_path)
        rendering_name = os.path.join(render_path, cam + "_" + image_name)
        gt_name = os.path.join(gts_path, cam + "_" + image_name)
        torchvision.utils.save_image(rendering, rendering_name)
        torchvision.utils.save_image(gt, gt_name)
        rendering = rendering[None].permute(0,2,3,1)
        gt = gt[None].permute(0,2,3,1).cuda()
        psnrs.append(eval_psnr(rendering, gt))
        ssims.append(eval_ssim(rendering, gt))
        lpips.append(eval_lpips(rendering, gt))

    outdict = {
        'psnr_mean': np.mean(psnrs),
        'psnrs': psnrs,
        'ssim_mean': np.mean(ssims),
        'ssims': ssims,
        'lpips_mean': np.mean(lpips),
        'lpips': lpips,
    }
    with open(os.path.join(output_path, name, "metrics.json"), 'w') as f:
        json.dump(outdict, f, indent=4)

def render_sets(dataset : ModelParams, time_duration : list, pipeline : PipelineParams, output):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False, time_duration=time_duration, load_iteration=30000)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #     resnder_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        
        # if not skip_test:
        render_set(output, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--output", default='out', type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_prue_model", default=False, type=bool)
    args = get_combined_args(parser)
    cfg = OmegaConf.load(args.config)
    args.time_duration = cfg.time_duration
    args.source_path = cfg.ModelParams.source_path
    args.env_map_res = cfg.PipelineParams.env_map_res
    args.output = os.path.join(args.output, os.path.basename(args.config).split(".")[0])
    print(args)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    model = model.extract(args)
    pipeline = pipeline.extract(args)
    if args.save_prue_model:
        os.makedirs(args.output, exist_ok=True)
        pth = torch.load(model.loaded_pth)
        print(f'iters: {pth[1]}')
        ckpt = pth[0]
        (active_sh_degree, 
        _xyz, 
        _features_dc, 
        _features_rest,
        _scaling, 
        _rotation, 
        _opacity,
        max_radii2D, 
        xyz_gradient_accum, 
        t_gradient_accum,
        denom,
        opt_dict, 
        spatial_lr_scale,
        _t,
        _scaling_t,
        _rotation_r,
        rot_4d,
        env_map,
        active_sh_degree_t) = ckpt
        needed = {
            '_xyz': _xyz,
            '_features_dc': _features_dc,
            '_features_rest': _features_rest,
            '_scaling': _scaling,
            '_rotation': _rotation,
            '_opacity': _opacity,
            '_t': _t,
            '_scaing_t': _scaling_t,
            '_rotation_r': _rotation_r,
            'env_map': env_map,
        }
        save_compressed = True
        if save_compressed:
            for k, v in needed.items():
                needed[k] = v.detach().cpu().numpy()
            np.savez_compressed(os.path.join(args.output, "pure_model.npz"), **needed)
        else:
            torch.save(needed, os.path.join(args.output, "pure_model.pth"))
    else:
        render_sets(model, args.time_duration, pipeline, args.output)