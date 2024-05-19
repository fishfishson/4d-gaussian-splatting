import os
from os.path import join
import argparse
import numpy as np
import json
import imagesize

from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # TODO: refine it.
    parser.add_argument("path", help="input path to the video")
    parser.add_argument("--fps", default=30, type=int, help="fps of the video")
    parser.add_argument("--frame_sample", default=[0,300,1], type=float, nargs=3)
    parser.add_argument("--test_views", default=[], nargs='+', help='test views')
    parser.add_argument("--camera_dir", default='optimized', type=str, help='camera directory')
    parser.add_argument("--include_test", action='store_true', default=False)
    args = parser.parse_args()

    # extract cameras
    print(f'read camera from {join(args.path, args.camera_dir)}')
    cameras = read_camera(join(args.path, args.camera_dir))
    camera_names = list(cameras.keys())
    camera_names = sorted(camera_names)
    print(camera_names)

    images = os.listdir(join(args.path, 'images', camera_names[0]))
    images = sorted(images)[args.frame_sample[0]:args.frame_sample[1]:args.frame_sample[2]]
    N = len(images)
    print(N)

    train_frames = []
    test_frames = []
    for k in tqdm(camera_names):
        K = cameras[k]['K'].copy()
        R = cameras[k]['R'].copy()
        T = cameras[k]['T'].copy()
        
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3:] = T
        w2c = w2c[:3]

        for i in tqdm(range(N)):
            image_path = join(args.path, 'images', k, images[i])
            frame = int(images[i].split('.')[0])
            W, H = imagesize.get(image_path)
            fl_x = K[0, 0]
            fl_y = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            cam_frames = [{
                'w': W,
                'h': H,
                'fl_x': fl_x,
                'fl_y': fl_y,
                'cx': cx,
                'cy': cy,
                'file_path': join('images', k, os.path.splitext(images[i])[0]), 
                'transform_matrix': w2c.tolist(),
                'time': frame / args.fps
            }]
            if args.include_test:
                if k in args.test_views:
                    test_frames += cam_frames
                train_frames += cam_frames
            else:
                if k in args.test_views:
                    test_frames += cam_frames
                else:
                    train_frames += cam_frames

    train_transforms = {
        'frames': train_frames,
    }
    test_transforms = {
        'frames': test_frames,
    }

    train_output_path = os.path.join(args.path, 'transforms_train.json')
    test_output_path = os.path.join(args.path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    