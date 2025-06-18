
import sys
import os
current_file_dir = os.path.dirname(__file__)

import glob
import cv2
workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)

import torch
import pypose as pp

import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from vis.utils.colmap import Dataset, Parser
from vis.utils.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from vis.utils.misc import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from vis.utils.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

def rotate_quat(points, rot_vecs):
    """
    Rotate 3D points by a pypose SE3 (tx, ty, tz, qw, qx, qy, qz).
    `rot_vecs` can be shape (7,) for a single transform or (B,7) for batches.
    """
    transform = pp.SE3(rot_vecs)
    return transform.Act(points)

def project(points, camera_params, camera_model=None):
    """
    Projects 3D points in world frame into 2D pixel coordinates using
    the final camera_params = [tx, ty, tz, qw, qx, qy, qz, fx, cx, cy].

    This sample assumes:
       - A pinhole camera model with focal length fx = fy.
       - The principal point is at (cx, cy).
    """
    # 1) Transform points to camera coordinates (rotation + translation).
    points_cam = rotate_quat(points, camera_params[..., :7])  # shape: (N, 3)

    # 2) Divide the first two coords by the depth (3rd coord).
    xy = points_cam[..., :2]
    z  = points_cam[..., 2].unsqueeze(-1)
    proj_2d = xy / z  # shape: (N, 2)

    # 3) Apply focal length and principal point shift.
    fx = camera_params[..., -3].unsqueeze(-1)  # shape: (1,)
    cx = camera_params[..., -2].unsqueeze(-1)  # shape: (1,)
    cy = camera_params[..., -1].unsqueeze(-1)  # shape: (1,)

    proj_2d = proj_2d * fx + torch.cat([cx, cy], dim=-1)  # shape: (N, 2)
    return proj_2d


def reproject_simple_pinhole(points, camera_params):
    points_proj = rotate_quat(points, camera_params[..., :7])
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)  # add dimension for broadcasting
    f = camera_params[..., -3].unsqueeze(-1)
    pp = camera_params[..., -2:]
    points_proj = points_proj * f + pp
    return points_proj


import torch






def assign_colors_to_points(
    points,       # (N, 3) in world coordinates
    intrinsics,   # (3, 3), e.g. [[fx,  0, cx],
                                #              [ 0, fy, cy],
                                #              [ 0,  0,  1]]
    extrinsic_raw,    # (4, 4) transform from world -> camera
    image,         # (H, W, 3), color image
    features_dc, features_extra, opacities, scales, rots
) -> torch.Tensor:
    """
    For each 3D point, project it into the image. If the resulting pixel is
    within the image bounds and its color is non-zero, assign that color
    to the point. Otherwise the point color is (0, 0, 0).

    Returns:
        color_array: (N, 3) float Tensor of RGB colors.
                     Points out-of-bounds, behind the camera,
                     or landing on a zero-pixel remain (0,0,0).
    """
    device = points.device
    H, W = image.shape[:2]
    points=points.to(torch.float64)
    N = points.shape[0]
    color_array = torch.zeros((N, 3), dtype=image.dtype, device=device)
    extrinsic=extrinsic_raw.to(torch.float32)
    intrinsics=intrinsics.to(torch.float32)
    extrinsic_numpy=extrinsic.cpu().detach().numpy()
    # 1) Transform points to camera coordinates

    # world2cam = extrinsic_numpy
    # transform_mat = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ], dtype=np.float64)

    # points_np=points.cpu().detach().numpy()
    # points = points_np @ world2cam[:3, :3].T + world2cam[:3, 3:].T
    # points = points @ transform_mat[:3, :3].T + transform_mat[:3, 3:].T
    # points = points.astype(np.float64)

    points=torch.tensor(points).to('cuda').to(torch.float32)
    rots = torch.tensor(rots).to('cuda').to(torch.float32)
    scales = torch.tensor(scales).to('cuda').to(torch.float32)
    covars=torch.ones((points.shape[0],6)).to('cuda').to(torch.float32)
    proj_results = fully_fused_projection(
        points,
        covars,
        rots,
        scales,
        torch.linalg.inv(extrinsic),
        intrinsics,
        W,
        H,
    )

    radii,means,depth,_,_=proj_results
    px=means[0,:,0]
    py=means[0,:,1]
    # camera_params=torch.cat((torch.tensor(extrinsic[:3, 3]),
    #                                      torch.tensor(R.from_matrix(extrinsic_numpy[:3, :3]).as_quat()).to('cuda')))
    # points_cam=reproject_simple_pinhole(points, camera_params)
    #                   # shape: (N, 3)


    # points_cam_valid = points_cam.squeeze(-1)

    # # 3) Project using pinhole model
    # #    x' = fx * x/z + cx,    y' = fy * y/z + cy
    # intrinsics=intrinsics.view(3,3)
    # fx = intrinsics[0, 0]
    # fy = intrinsics[1, 1]
    # cx = intrinsics[0, 2]
    # cy = intrinsics[1, 2]

    # u = points_cam_valid[:, 0]
    # v = points_cam_valid[:, 1]




    # 4) Filter out projections that lie outside image bounds
    in_bounds = (
        (px >= 0) & (px < W) &
        (py >= 0) & (py < H)
    )
    px_in = px[in_bounds]
    py_in = py[in_bounds]
    in_bounds_indices = torch.where(in_bounds)[0]

    # 5) For each valid pixel, assign color only if the pixel is non-zero
    for i in range(px_in.shape[0]):
        x_i = int(px_in[i])
        y_i = int(py_in[i])
        pt_idx = in_bounds_indices[i]

        pixel_val = image[y_i, x_i]  # shape: (3,)
        
        color_array[pt_idx] = pixel_val

    return color_array

import numpy as np
from plyfile import PlyData,PlyElement

def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]), np.array(plydata.elements[0]["y"]), np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    # e.g. 3 * (max_sh_degree + 1) ^ 2 - 3  ->  3*(3+1)^2 - 3 = 3*16 - 3 = 48 - 3 = 45
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    # reshape to (num_points, 3, (#SHcoeffs except DC))
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    # scale
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.array(plydata.elements[0][attr_name])

    # rotation
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.array(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots

def load_heatmap(folder_path, save_path):
    """
    Load, rotate, convert to RGB, pad, and save images from 'folder_path'.
    Each image is rotated 90 degrees CCW and padded to 3840x1260 with blue background.
    Returns a list of NumPy arrays (RGB images).
    """
    exts = ("*.png", "*.jpg", "*.jpeg")
    heatmaps = []

    os.makedirs(save_path, exist_ok=True)

    for ext in exts:
        pattern = os.path.join(folder_path, ext)
        for file_path in sorted(glob.glob(pattern)):
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Warning: Could not read image: {file_path}")
                continue

            heatmaps.append(img)

    return heatmaps

    

import numpy as np

def save_ply(points, colors, out_path):
    """
    Save a colored point cloud to a .ply file (ASCII format).
    
    :param points: Nx3 NumPy array of 3D points.
    :param colors: Nx3 NumPy array of RGB color values (0-255).
    :param out_path: Output file path (string).
    """
    # Ensure shapes match
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of vertices."
    assert points.shape[1] == 3, "Points should be of shape (N, 3)."
    assert colors.shape[1] == 3, "Colors should be of shape (N, 3)."

    num_points = points.shape[0]

    # Create and write header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]

    with open(out_path, 'w') as f:
        # Write the header lines
        for line in header:
            f.write(line + "\n")

        # Write each point + color
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            # Convert color to int (if they are floats)
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    print(f"Saved PLY: {out_path}")




def vote_color_exclude_zeros(global_colors_list):
    """
    Performs a majority vote on the colors for each pixel with special handling:
    - If a pixel has more than 2 votes with high red values (R channel > 200), it is set to pure red ([255, 0, 0]).
    - If all M votes for a pixel are zeros, the result is zero.
    - If there is at least one non-zero color, all zero votes are ignored, and the majority vote
      is computed from the non-zero colors.

    Parameters:
        global_colors_list (np.ndarray): An array of shape (M, N, 3) where each [m, n, :]
                                         represents a color vote (e.g., RGB) for pixel n.

    Returns:
        voted_color (np.ndarray): An (N, 3) array with the voted color for each pixel.
    """
    M, N, _ = global_colors_list.shape
    voted_color = np.zeros((N, 3), dtype=global_colors_list.dtype)

    for n in range(N):
        votes = global_colors_list[:, n, :]
        
        # Check for votes with red channel significantly high
        red_votes = np.sum(votes[:, 0] > 200)
        if red_votes > 2:
            voted_color[n] = np.array([255, 0, 0], dtype=global_colors_list.dtype)
            continue

        non_zero_mask = ~np.all(votes == 0, axis=1)

        if np.any(non_zero_mask):
            votes = votes[non_zero_mask]

        if votes.shape[0] == 0:
            voted_color[n] = np.zeros(3, dtype=global_colors_list.dtype)
        else:
            unique_colors, counts = np.unique(votes, axis=0, return_counts=True)
            majority_index = np.argmax(counts)
            voted_color[n] = unique_colors[majority_index]

    return voted_color



import numpy as np

def rescale_red_green(voted_color, stddev=10, final_color_variation=50):
    """
    Remaps colors:
    - Red pixels (red dominant) are changed to green with random variation.
    - All other pixels are changed to red with random variation.

    Adds additional random variation to the final color output.

    Parameters:
        voted_color (np.ndarray): (N, 3) array of input colors (uint8).
        stddev (float): Standard deviation for added random noise.
        final_color_variation (float): Additional random variation for the final colors.

    Returns:
        np.ndarray: (N, 3) array with remapped colors.
    """
    result = voted_color.copy().astype(np.int32)

    # Define target colors
 # Define target colors
    target_cyanish_green = np.array([0, 255, 200], dtype=np.int32)
    target_red = np.array([255, 0, 0], dtype=np.int32)

    # Use this target instead of pure green
    # Mask for red dominance
    red_mask = (voted_color[:, 0] > voted_color[:, 1]) & (voted_color[:, 0] > voted_color[:, 2])
    non_red_mask = ~red_mask

    # Red pixels to green
    num_red = np.count_nonzero(red_mask)
    if num_red > 0:
        noise = np.random.normal(0, stddev, (num_red, 3))
        new_green_colors = np.clip(target_cyanish_green + noise, 0, 255)
        result[red_mask] = new_green_colors.astype(np.int32)

    # Non-red pixels to red
    num_non_red = np.count_nonzero(non_red_mask)
    if num_non_red > 0:
        noise = np.random.normal(0, stddev, (num_non_red, 3))
        new_red_colors = np.clip(target_red + noise, 0, 255)
        result[non_red_mask] = new_red_colors.astype(np.int32)

    # Additional final random variation
    final_noise = np.random.normal(0, final_color_variation, result.shape)
    result = np.clip(result + final_noise, 0, 255)

    # Convert back to uint8
    return result.astype(np.uint8)


# ----------------------
# Example Usage
if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm
    # Suppose we have:
    # 1) A random set of 3D points (N,3)
    # 2) Camera intrinsics (3,3)
    # 3) A random image (H,W,3)
    # 4) An SE3 camera extrinsic in pypose

    import pypose as pp


    # load points the gaussian ply, but only keep the means
    # points_world, features_dc, features_extra, opacities, scales, rots = load_ply('/home/haozhe/Dropbox/thermalhand/0524/iron_20250511_1717/object.ply')
    # points_world, features_dc, features_extra, opacities, scales, rots = load_ply('/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650/shortcanpoints.ply')

    # factor=1
    # # data_dir = '/home/haozhe/Dropbox/thermalhand/0524/iron_20250511_1717/result'
    # data_dir = '/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650/result'
    # # output_dir=
    # parser=Parser(
    #     data_dir=data_dir, factor=factor, normalize=True, test_every=8
    # )
    # valset= Dataset(parser, split="train")
    # valloader = torch.utils.data.DataLoader(
    #         valset, batch_size=1, shuffle=False, num_workers=1
    #     )
    # images= load_heatmap('/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650/output_images',save_path='/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650/processed_thermal')


    # global_colors_list=[]
    # for i, data in enumerate(valloader):
    #         camtoworlds = data["camtoworld"]

    #         # camtoworlds[:3,3]= camtoworlds[:3,3]+ np.array([0.00,-0.003,-0.08,0])
    #         # 4*4 camtoworlds

            
    #         Ks = data["K"]
    #         # Ks=torch.tensor([[1204, 0, 640],
    #         #             [0, 1019, 480],
    #         #             [0, 0, 1]], dtype=torch.float64).view(1,3,3)

  

    #         image=images[i]
    #         savename='/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650'
    # # Fake color image
    # # shape = (H, W, 3)
    # # For demonstration, fill it with random values or any valid image
    # # image = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8)


    # # load the image from this path: the format is that each cap means the image id, load the image in this cap id, and 
    # # save it to the image list
    #         points_world_torch= torch.from_numpy(points_world).to('cuda')
    #         # Ks=torch.from_numpy(Ks).to('cuda')
    #         image_torch= torch.from_numpy(image).to('cuda')
    # # Now run our function
    #         point_colors = assign_colors_to_points(points_world_torch, Ks.to('cuda'), camtoworlds.to('cuda'), image_torch,features_dc, features_extra, opacities, scales, rots)
    #         colors=point_colors.cpu().detach().numpy()
    #         global_colors_list.append(colors)


    #         # Save the colored point cloud in PLY format
    
    
    
    #         # o3d.io.write_point_cloud(os.path.join(savename,"colored_cloud.ply"), pcd)

    # #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_world)

    # voted_color=vote_color_exclude_zeros(np.asarray(global_colors_list))

    # Load point cloud from PLY file
    pcd = o3d.io.read_point_cloud('/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650/colored_cloud_final_raw_K.ply')

    # Ensure color data is scaled to [0, 255] uint8 for processing
    original_colors = np.asarray(pcd.colors) * 255
    original_colors = original_colors.astype(np.uint8)

    # Apply color rescaling function
    final_colors = rescale_red_green(original_colors)

    # Rescale colors back to [0, 1] for Open3D
    final_colors_normalized = final_colors.astype(np.float64) / 255.0

    # Assign new colors back to point cloud
    pcd.colors = o3d.utility.Vector3dVector(final_colors_normalized)

    # Save the modified point cloud
    savename = '/home/haozhe/Dropbox/thermalhand/0524/short_hot_can_20250511_1650'
    output_path = os.path.join(savename, "heat.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    print("Final assigned colors for each 3D point saved to:", output_path)