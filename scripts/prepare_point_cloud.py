import sys
sys.path.append('.')
import os
import numpy as np
from scene.dataset_readers import fetchPly
import argparse
from plyfile import PlyData, PlyElement

def storePly(path, xyz, rgb, times):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('time', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, times), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare point cloud')
    parser.add_argument('--ply_dir', type=str, default='data/dvv/11_Alexa_Meade_Face_Paint_2/pcds')
    parser.add_argument('--output', type=str, default='data/dvv/11_Alexa_Meade_Face_Paint_2/4dgs_init.ply')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    ply_dir = args.ply_dir

    xyzs = []
    rgbs = []
    normals = []
    times = []
    interval = 1 / args.fps
    for ply_name in sorted(os.listdir(ply_dir)):
        if ply_name.endswith('.ply'):
            ply_path = os.path.join(ply_dir, ply_name)
        else:
            import ipdb; ipdb.set_trace()
        frame_ply = fetchPly(ply_path)
        frame_timestamp = float(ply_name.split('.')[0]) * interval
        
        xyzs.append(frame_ply.points)
        rgbs.append(frame_ply.colors)
        # normals.append(frame_ply.normals)
        times.append(np.full((frame_ply.points.shape[0], 1), frame_timestamp))
    
    xyzs = np.concatenate(xyzs)
    rgbs = np.concatenate(rgbs)
    # normals = np.concatenate(normals)
    times = np.concatenate(times)
    
    storePly(args.output, xyzs, rgbs, times)
    
    


