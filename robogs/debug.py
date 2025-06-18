# this is the api for debug the coordinate, scale, mdh parameter and alignment.

from runner import Runner
from mdh import *
from assign import *
from mesh_util.align import *

from render_util import filter_with_semantic_torch,filter_with_semantic
import mujoco as mj
import numpy as np
import torch
import open3d as o3d
from assign import load_ply_sam,save_ply_sam,load_ply
from deform_util import sh_rotation_torch, quaternion_to_matrix, matrix_to_quaternion
from gsplat.rendering import rasterization
from torch import Tensor

import sys
import os
current_file_dir = os.path.dirname(__file__)


workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)

from robogs.engine.mdh.mdh2 import calculate_franka_mdh_pre_frame,inverse_affine_transformation_torch,calculate_franka_with_gripper_mdh_pre_frame
import mujoco.viewer as mv
import mujoco
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
from gsplat.strategy import DefaultStrategy, MCMCStrategy



scene_translation= np.array([0.0, 0.0, 0.0])
scene_rotation = np.array([0.0, 0.0, 0.0])  # in radians



@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Close viewer after training
    close_viewer_after_training: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)




def sync_action(mj_model, mj_data, action, robot_action):
    """
    Syncs the action between the Mujoco model and the robot.
    """
    mj_data.ctrl[:7] = robot_action
    mj_data.ctrl[7:] = action
    mj.mj_step(mj_model, mj_data)
    mj_data.update()



def _append_outputs( outputs, xyz, opacities, scales, features_extra, rotation, features_dc, semantic_id):
        """Append values to output lists."""
        outputs["xyz"].append(xyz)
        outputs["opacities"].append(opacities) 
        outputs["scales"].append(scales)
        outputs["features_extra"].append(features_extra)
        outputs["rots"].append(rotation)
        outputs["features_dc"].append(features_dc)
        outputs["semantic_id"].append(semantic_id)


flip_matrix = torch.tensor([
    [1,  0,  0],
    [0,  1,  0],
    [0, 0,  -1]
]).type(torch.float32)


def main(cfg,file_path='/home/louhz/Desktop/ucb/asset/final_scene_with_ids.ply'):
    # 1. Load the MuJoCo model
    # mj_file = '/home/haozhe/Dropbox/physics/franka_leap_demo/scene_ketchup_render.xml'
    # mj_model = mj.MjModel.from_xml_path(mj_file)
    # mj_data = mj.MjData(mj_model)


    # # setup object mass:
    # mj_data.model.body_mass[mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, 'ketchup')] = 1.5
    # mj_data.model.body_mass[mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, 'ketchup')] = 0.7

    # 2. Load the SAM-labeled .ply file (from GroundingSAM or a similar pipeline)
    ply_file = file_path
    xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = load_ply_sam(ply_file)

    worldscale=torch.ones(3)
    worldscale[0]=1
    worldscale[1]=1
    worldscale[2]=1

    # also a coordinate issue

        # just query from body  based on name
    # id list
    class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    class_name= [
        (("link0"), 1),
        (("link1"), 2),
        (("link2"), 3),
        (("link3"), 4),
        (("link4"), 5),
        (("link5"), 6),
        (("link6"), 7),
        (("link7"), 8),
        (("gripper_main"), 9),
        (("gripper_left1"), 10),
        (("gripper_left2"), 11),
        (("gripper_right1"), 12),
        (("gripper_right2"), 13),
        # (("tablemesh"), 14),
        (("cup"), 14),
    ]


    body_names=['attachment', 'ketchup', 'if_bs', 'if_ds', 'if_md', 'if_px',
                 'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                   'mf_bs', 'mf_ds', 'mf_md', 'mf_px',
                     'palm', 'plate', 
                     'rf_bs', 'rf_ds', 'rf_md', 'rf_px', 'th_bs',
                       'th_ds', 'th_mp', 'th_px', 'world']




    outputs = {
            "xyz": [], "opacities": [], "scales": [], "features_extra": [],
            "rots": [], "features_dc": [], "semantic_id": []
        }
    
    active_id=[2,3,4,5,6,7,8,9,10,11,12,13]

    # find 15 recenter vector the the object

    # query for the pos diff and quat diff

    # perform the forward transformation

    recenter_list=np.zeros((len(class_ids), 3))
    recenter_list[0]= np.array([0, 0, 0])
    recenter_list[1]= np.array([0, 0, 0])
    # arm
    recenter_list[2]= np.array([-0.068, -0.3, 0.025])
    recenter_list[3]= np.array([-0.1, -0.568, 0.084])
    recenter_list[4]= np.array([-0.21, -0.791, 0.042])
    recenter_list[5]= np.array([-0.285, -1, -0.044])
    recenter_list[6]= np.array([-0.0612, -1.118, -0.157])
    recenter_list[7]= np.array([-0.858, -0.95, -0.248])
    recenter_list[8]= np.array([-0.9, -0.937, -0.121])
    #hand
    recenter_list[9]= np.array([-0.916, -0.828, -0.044])
    recenter_list[10]= np.array([-1.058, -0.848, 0])
    recenter_list[11]= np.array([-1.0, -0.66, -0.04])
    recenter_list[12]= np.array([0.933, 0.632, -0.054])
    recenter_list[13]= np.array([0.842, 0.624, -0.053])
    recenter_list[14]= np.array([-0.943, -0.179, 0.185])

    inv_joint_value = torch.tensor([0.1448, -0.1007, -0.0946, -2.3859, 0.0028, 2.1279, 0.034,
                                    0,
                                    0,0,0,0,0], dtype=torch.float)
                                    
    current_joint_value= torch.tensor([0, 0, 0.2946, -1.5859, 0.4028, 1.3279, 2.034,
                                    0,
                                    0,0,0,0,0], dtype=torch.float)
                         


    background_id=[1,14]
    base_id=[0,1]
    for index in range(len(background_id)):
            select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
                semantic_id,
                background_id,
                xyz,
                opacities,
                scales,
                features_extra,
                rots,
                features_dc,
                index) 
            
            select_xyz= torch.tensor(select_xyz, dtype=torch.float)
            select_opacities       = torch.tensor(select_opacities,  dtype=torch.float)
            select_scales          = torch.tensor(select_scales,     dtype=torch.float)
            select_features_extra  = torch.tensor(select_features_extra, dtype=torch.float)
            select_rotation        = torch.tensor(select_rotation, dtype=torch.float)
            select_feature_dc      = torch.tensor(select_feature_dc, dtype=torch.float)
            semantic_id_ind_sam  = torch.tensor(semantic_id_ind_sam, dtype=torch.float)
            
            _append_outputs(outputs, select_xyz, select_opacities, select_scales, 
                            select_features_extra, select_rotation, 
                            select_feature_dc, semantic_id_ind_sam)
    for index in range(len(active_id)):
            # Filter the relevant data for this semantic_id
            (select_xyz, 
            select_opacities, 
            select_scales, 
            select_features_extra, 
            select_rotation, 
            select_feature_dc, 
            semantic_id_ind_sam) = filter_with_semantic(
                semantic_id,
                active_id,
                xyz,
                opacities,
                scales,
                features_extra,
                rots,
                features_dc,
                index
            )
            
            select_opacities       = torch.tensor(select_opacities,  dtype=torch.float)
            select_scales          = torch.tensor(select_scales,     dtype=torch.float)
            select_feature_dc      = torch.tensor(select_feature_dc, dtype=torch.float)
            semantic_id_ind_sam_t  = torch.tensor(semantic_id_ind_sam, dtype=torch.float)
            
            mark_id = active_id[index]
            
            # Center vector is zero in your snippet, but adapt if needed:
            center_vector_gt = np.zeros((3,))
            raw_xyz_centered = torch.from_numpy(select_xyz + center_vector_gt).float()
            
            # Compute inverse transform from inv_joint_value
            # (Focus only on the first 9 joints in your example if needed)
            _, inv_transformation = calculate_franka_with_gripper_mdh_pre_frame(inv_joint_value[:14])
            inv_trans = inverse_affine_transformation_torch(inv_transformation[:14])
            
            # Kinematic index = mark_id - 2 (based on your snippet)
            kinematic_id = mark_id - len(base_id)
            
            inv_rotation_raw = inv_trans[kinematic_id][0:3, 0:3]
            inv_translation  = inv_trans[kinematic_id][0:3, 3]
            
            # Forward transform for current_joint_value
            _, transformation = calculate_franka_with_gripper_mdh_pre_frame(current_joint_value[:14])
            rotation_raw      = transformation[kinematic_id][0:3, 0:3]
            translation       = transformation[kinematic_id][0:3, 3]
            
            # 1) "Undo" the old transform
            deform_point = raw_xyz_centered @ inv_rotation_raw.T + inv_translation
            # 2) Apply the new transform
            forward_point = deform_point @ rotation_raw.T + translation
            select_xyz    = forward_point
            
            # Combine rotations for any orientation-based features
            rotation_splat = rotation_raw @ inv_rotation_raw
            
            # Deform the orientation quaternions
            rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
            rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
            select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)
            
            # Rotate your Spherical Harmonic (SH) features
            select_features_extra_deform = sh_rotation_torch(
                torch.tensor(select_features_extra, dtype=torch.float),
                select_feature_dc,
                rotation_splat
            )
            
            # Accumulate everything to outputs
            _append_outputs(
                outputs, 
                select_xyz,
                select_opacities,
                select_scales,
                select_features_extra_deform,
                select_rotation_deform,
                select_feature_dc,
                semantic_id_ind_sam_t
            )
    # finger_id1=[11]



    # for index in range(len(finger_id1)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id1,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        

    #     mark_id=finger_id1[index]
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)

    #     recenter_vector=np.array([-0.945,-0.613,-0.05])
    #      # 2) Apply recenter vector
    #     #    "Recenter" means translate so that recenter_vector is at origin
    #     select_xyz_recentered = torch.from_numpy(np.array(select_xyz + recenter_vector)).to(torch.float)

    #     # q=torch.tensor([-10]).deg2rad() # q_11_1
    #     # q=torch.tensor([-20]).deg2rad() # q_11_2
    #     q=torch.tensor([-30]).deg2rad() # q_11_3
    #     # q=torch.tensor([-40]).deg2rad() # q_11_4


    #     rot=torch.tensor([
    #         [1, 0, 0],
    #         [0, torch.cos(q), torch.sin(q)],
    #         [0, -torch.sin(q), torch.cos(q)]
    #     ])


    #     rotation_splat = rot # [3 x 3]

    #     # translation_offset = torch.tensor([0.052, -0.374, -0.105], dtype=torch.float) # t_11_0
    #     # translation_offset = torch.tensor([0.052, -0.374, -0.115], dtype=torch.float) # t_11_1
    #     translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_11_2
    #     # translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_11_3



    #     select_xyz_deformed = select_xyz_recentered@rotation_splat.T
    #     select_xyz_deformed = select_xyz_deformed - recenter_vector  + translation_offset


    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )





    #     _append_outputs(outputs, select_xyz_deformed, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)


    # finger_id2=[12]



    # for index in range(len(finger_id2)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id2,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        

    #     mark_id=finger_id2[index]
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)

    #     recenter_vector=np.array([-0.926,-0.609,-0.06])
    #      # 2) Apply recenter vector
    #     #    "Recenter" means translate so that recenter_vector is at origin
    #     select_xyz_recentered = torch.from_numpy(np.array(select_xyz + recenter_vector)).to(torch.float)
        

    #     # q=torch.tensor([-10]).deg2rad() # q_12_1
    #     # q=torch.tensor([-20]).deg2rad() # q_12_2
    #     q=torch.tensor([-30]).deg2rad() # q_12_3
    #     # q=torch.tensor([-40]).deg2rad() # q_12_4
    #     rot=torch.tensor([
    #         [1, 0, 0],
    #         [0, torch.cos(q), torch.sin(q)],
    #         [0, -torch.sin(q), torch.cos(q)]
    #     ])


    #     rotation_splat = rot # [3 x 3]

    #     # translation_offset = torch.tensor([0.052, -0.364, -0.105], dtype=torch.float) # t_12_0
    #     # translation_offset = torch.tensor([0.052, -0.364, -0.115], dtype=torch.float) # t_12_1
    #     translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_12_2
    #     # translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_12_3



    #     select_xyz_deformed = select_xyz_recentered@rotation_splat.T
    #     select_xyz_deformed = select_xyz_deformed - recenter_vector  + translation_offset


    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )





    #     _append_outputs(outputs, select_xyz_deformed, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)
        

    # finger_id3=[13]



    # for index in range(len(finger_id3)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id3,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        

    #     mark_id=finger_id3[index]
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)

    #     recenter_vector=np.array([-0.852,-0.616,-0.06])
    #      # 2) Apply recenter vector
    #     #    "Recenter" means translate so that recenter_vector is at origin
    #     select_xyz_recentered = torch.from_numpy(np.array(select_xyz + recenter_vector)).to(torch.float)
        

    #     # q=torch.tensor([-10]).deg2rad() # q_13_1
    #     # q=torch.tensor([-20]).deg2rad() # q_13_2
    #     q=torch.tensor([-30]).deg2rad() # q_13_3
    #     # q=torch.tensor([-40]).deg2rad() # q_13_4

    #     rot=torch.tensor([
    #         [1, 0, 0],
    #         [0, torch.cos(q), torch.sin(q)],
    #         [0, -torch.sin(q), torch.cos(q)]
    #     ])


    #     rotation_splat = rot # [3 x 3]


    #     # translation_offset = torch.tensor([0.052, -0.320, -0.117], dtype=torch.float) # t_13_0
    #     # translation_offset = torch.tensor([0.052, -0.320, -0.127], dtype=torch.float) # t_13_1
    #     translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_13_2
    #     # translation_offset = torch.tensor([0.052, -0.334, -0.135], dtype=torch.float) # t_13_3


    #     select_xyz_deformed = select_xyz_recentered@rotation_splat.T
    #     select_xyz_deformed = select_xyz_deformed - recenter_vector  +translation_offset


    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )





    #     _append_outputs(outputs, select_xyz_deformed, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)
        
   
    # finger_id1=[11]



    # for index in range(len(finger_id1)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id1,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)
    #        # center_vector_gt = flip_matrix.numpy()@pos[mark_id]* worldscale.numpy()
    #     # center_vector_gt=recenter_list[mark_id]
    #     center_vector_gt = np.zeros((3))
    #     raw_xyz_centered = torch.from_numpy(select_xyz + center_vector_gt).to(torch.float)
        
    #     kinematic_id=7
        
    #     inv_transformation_raw, inv_transformation= calculate_franka_mdh_pre_frame(inv_joint_value[:9])


    #     inv_trans=inverse_affine_transformation_torch(inv_transformation[:9])
        
    #     inv_rotation_raw=inv_trans[kinematic_id][0:3, 0:3]
    #     inv_translation=inv_trans[kinematic_id][0:3, 3]


    #     inv_rotation=inv_rotation_raw
    #     transformation_raw,transformation= calculate_franka_mdh_pre_frame(joint_control_action)
    #     rotation_raw=transformation[kinematic_id][0:3, 0:3]
    #     translation=transformation[kinematic_id][0:3, 3]

    #     rotation=rotation_raw


    #     deform_point= raw_xyz_centered @ inv_rotation.T+ inv_translation 

    #     forward_point=  deform_point @ rotation.T+ translation 
    #     select_xyz = forward_point

    #     # # Combine rotations (example: rotation_splat = rotation * rotation_inv)
    #     rotation_splat = rotation @ inv_rotation

    #     # Convert your chosen quaternion to a matrix, multiply, convert back:
    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     # Rotate your SH or other feature embeddings
    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )




    #     _append_outputs(outputs, select_xyz, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)


    # finger_id2=[12]



    # for index in range(len(finger_id2)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id2,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)
        
    #     center_vector_gt = np.zeros((3))
    #     raw_xyz_centered = torch.from_numpy(select_xyz + center_vector_gt).to(torch.float)
        
    #     kinematic_id=7
        
    #     inv_transformation_raw, inv_transformation= calculate_franka_mdh_pre_frame(inv_joint_value[:9])


    #     inv_trans=inverse_affine_transformation_torch(inv_transformation[:9])
        
    #     inv_rotation_raw=inv_trans[kinematic_id][0:3, 0:3]
    #     inv_translation=inv_trans[kinematic_id][0:3, 3]


    #     inv_rotation=inv_rotation_raw
    #     transformation_raw,transformation= calculate_franka_mdh_pre_frame(joint_control_action)
    #     rotation_raw=transformation[kinematic_id][0:3, 0:3]
    #     translation=transformation[kinematic_id][0:3, 3]

    #     rotation=rotation_raw


    #     deform_point= raw_xyz_centered @ inv_rotation.T+ inv_translation 

    #     forward_point=  deform_point @ rotation.T+ translation 
    #     select_xyz = forward_point

    #     # # Combine rotations (example: rotation_splat = rotation * rotation_inv)
    #     rotation_splat = rotation @ inv_rotation

    #     # Convert your chosen quaternion to a matrix, multiply, convert back:
    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     # Rotate your SH or other feature embeddings
    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )




    #     _append_outputs(outputs, select_xyz, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)

        

    # finger_id3=[13]



    # for index in range(len(finger_id3)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         finger_id3,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        
    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)
    #     center_vector_gt = np.zeros((3))
    #     raw_xyz_centered = torch.from_numpy(select_xyz + center_vector_gt).to(torch.float)
        
    #     kinematic_id=7
        
    #     inv_transformation_raw, inv_transformation= calculate_franka_mdh_pre_frame(inv_joint_value[:9])


    #     inv_trans=inverse_affine_transformation_torch(inv_transformation[:9])
        
    #     inv_rotation_raw=inv_trans[kinematic_id][0:3, 0:3]
    #     inv_translation=inv_trans[kinematic_id][0:3, 3]


    #     inv_rotation=inv_rotation_raw
    #     transformation_raw,transformation= calculate_franka_mdh_pre_frame(joint_control_action)
    #     rotation_raw=transformation[kinematic_id][0:3, 0:3]
    #     translation=transformation[kinematic_id][0:3, 3]

    #     rotation=rotation_raw


    #     deform_point= raw_xyz_centered @ inv_rotation.T+ inv_translation 

    #     forward_point=  deform_point @ rotation.T+ translation 
    #     select_xyz = forward_point

    #     # # Combine rotations (example: rotation_splat = rotation * rotation_inv)
    #     rotation_splat = rotation @ inv_rotation

    #     # Convert your chosen quaternion to a matrix, multiply, convert back:
    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     # Rotate your SH or other feature embeddings
    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )




    #     _append_outputs(outputs, select_xyz, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
    #                     select_feature_dc, semantic_id_ind_sam)




    # object_ids=[15]

    # # can you write a code that iterate over the step and 
    # # try to save every 10 step of the object ply with the relative step 
    # step=190
    # for index in range(len(object_ids)):
    #     select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam=filter_with_semantic(
    #         semantic_id,
    #         object_ids,
    #         xyz,
    #         opacities,
    #         scales,
    #         features_extra,
    #         rots,
    #         features_dc,
    #         index) 
        

    #     select_opacities=torch.from_numpy(select_opacities).to(torch.float)
    #     select_scales=torch.from_numpy(select_scales).to(torch.float)
    #     select_feature_dc=torch.from_numpy(select_feature_dc).to(torch.float)
    #     semantic_id_ind_sam=torch.from_numpy(semantic_id_ind_sam).to(torch.float)

    #     mark_id=object_ids[index]

    #     recenter_vector=recenter_list[15]

    #     select_xyz_recentered = torch.tensor(select_xyz + recenter_vector).to(torch.float)
        
    #     scale=np.array([1, 1, 1])




    #     # q=torch.tensor([0]).deg2rad() # q0 -
    #     # q=torch.tensor([10]).deg2rad() # q1 -
    #     # q=torch.tensor([20]).deg2rad() # q2 -
    #     # q=torch.tensor([30]).deg2rad() # q3 
    #     # q=torch.tensor([40]).deg2rad() # q4 -
    #     # q=torch.tensor([50]).deg2rad() # q5 -
    #     # q=torch.tensor([60]).deg2rad() # q6 -
    #     # q=torch.tensor([70]).deg2rad() # q7 -
    #     # q=torch.tensor([75]).deg2rad() # q8 -
    #     # q=torch.tensor([80]).deg2rad() # q9 -
    #     q=torch.tensor([85]).deg2rad() # q10 


    #     rot=torch.tensor([
    #         [1, 0, 0],
    #         [0, torch.cos(q), torch.sin(q)],
    #         [0, -torch.sin(q), torch.cos(q)]
    #     ])

    #     rotation_splat = rot # [3 x 3]
        




    #     # translation_offset = torch.tensor([0, 0.055, -0.13], dtype=torch.float) # Basic Standard
    #     # translation_offset = torch.tensor([0, -0.00, -0.00], dtype=torch.float) # t0 -> q0 -
    #     # translation_offset = torch.tensor([0, -0.000, -0.02], dtype=torch.float) # t1 -> q1 -
    #     # translation_offset = torch.tensor([0, -0.002, -0.04], dtype=torch.float) # t2 -> q2 -
    #     # translation_offset = torch.tensor([0, -0.0035, -0.06], dtype=torch.float) # t3 -> q3 -
    #     # translation_offset = torch.tensor([0, -0.0062, -0.08], dtype=torch.float) # t4 -> q4 -
    #     # translation_offset = torch.tensor([0, -0.020, -0.10], dtype=torch.float) # t5 -> q5 -
    #     # translation_offset = torch.tensor([0, -0.037, -0.115], dtype=torch.float) # t6 -> q6 -
    #     # translation_offset = torch.tensor([0, -0.055, -0.125], dtype=torch.float) # t7 -> q7 -
    #     # translation_offset = torch.tensor([0, -0.090, -0.135], dtype=torch.float) # t8 -> q8 -
    #     # translation_offset = torch.tensor([0, -0.105, -0.140], dtype=torch.float) # t9 -> q9 -
    #     translation_offset = torch.tensor([0, -0.115, -0.142], dtype=torch.float) # t10 -> q10
               
        
        
        
        
        
        
    #     select_xyz_deformed = select_xyz_recentered@rotation_splat.T
    #     select_xyz_deformed = select_xyz_deformed - recenter_vector +translation_offset


    #     rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
    #     rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
    #     select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)

    #     select_features_extra_deform = sh_rotation_torch(
    #         torch.tensor(select_features_extra, dtype=torch.float),
    #         torch.tensor(select_feature_dc, dtype=torch.float),
    #         rotation_splat
    #     )



    #     _append_outputs(outputs, select_xyz_deformed, select_opacities, select_scales, 
    #                     select_features_extra_deform, select_rotation_deform, 
                        # select_feature_dc, semantic_id_ind_sam)
    


    xyz= torch.cat(outputs["xyz"], dim=0).cpu().numpy()
    opacities= torch.cat(outputs["opacities"], dim=0).cpu().numpy()   
    scales= torch.cat(outputs["scales"], dim=0).cpu().numpy()
    features_extra= torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
    rots= torch.cat(outputs["rots"], dim=0).cpu().numpy()
    features_dc= torch.cat(outputs["features_dc"], dim=0).cpu().numpy()
    semantic_id= torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()

    # or render it here



    save_ply_sam(
        xyz=xyz,
        opacities=opacities,
        scale=scales,
        f_rest=features_extra,
        rotation=rots,  
        f_dc=features_dc,
        semantic_id=semantic_id,
        path='/home/louhz/Desktop/ucb/asset/debug/t.ply',

    )
    # render for the new deformed gs



def activate_parser(cfg):
    parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
    )

    return parser





if __name__ == '__main__':
    # Parse the configuration from command-line.
    configs = {
        "default": (
            "Gaussian splatting extraction for novel view and editing.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    main(cfg)
    # parser = Parser(
    #         data_dir=cfg.data_dir,
    #         factor=cfg.data_factor,
    #         normalize=cfg.normalize_world_space,
    #         test_every=cfg.test_every,
    #     )
    # local_rank=0
    # world_rank=0
    # world_size=1
    # runner = Runner(local_rank, world_rank, world_size, cfg)
    # if cfg.ckpt is not None:
    #     # run eval only
    #     ckpts = [
    #         torch.load(file, map_location=runner.device, weights_only=True)
    #         for file in cfg.ckpt
    #     ]
    #     for k in runner.splats.keys():
    #         runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
    #     step = ckpts[0]["step"]
    #     runner.eval(step=step,ply_file='/home/haozhe/Dropbox/rendering/asset/final_scene_with_ids.ply')

        # load the final_scene_with_ids, edit it

        # then record the time frame

        # runner.exportdeform(step=step,ply_file='/home/haozhe/Dropbox/rendering/asset/final_scene_with_ids.ply')
        # runner.render_traj(step=step)
    # renderer(cfg,parser=parser)