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

from mdh import calculate_franka_mdh_pre_frame,inverse_affine_transformation_torch
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


def _append_outputs( outputs, xyz, opacities, scales, features_extra, rotation, features_dc, semantic_id):
        """Append values to output lists."""
        outputs["xyz"].append(xyz)
        outputs["opacities"].append(opacities) 
        outputs["scales"].append(scales)
        outputs["features_extra"].append(features_extra)
        outputs["rots"].append(rotation)
        outputs["features_dc"].append(features_dc)
        outputs["semantic_id"].append(semantic_id)



recenter_list=np.zeros((15, 3))
# recenter_list[0]= np.array([0, 0, 0])
# recenter_list[1]= np.array([0, 0, 0])
#     # arm
#     recenter_list[2]= np.array([-0.068, -0.3, 0.025])
#     recenter_list[3]= np.array([-0.1, -0.568, 0.084])
#     recenter_list[4]= np.array([-0.21, -0.791, 0.042])
#     recenter_list[5]= np.array([-0.285, -1, -0.044])
#     recenter_list[6]= np.array([-0.0612, -1.118, -0.157])
#     recenter_list[7]= np.array([-0.858, -0.95, -0.248])
#     recenter_list[8]= np.array([-0.9, -0.937, -0.121])
#     #hand
#     recenter_list[9]= np.array([-0.916, -0.828, -0.044])
#     recenter_list[10]= np.array([-1.058, -0.848, 0])
#     recenter_list[11]= np.array([-1.0, -0.66, -0.04])
#     recenter_list[12]= np.array([0.933, 0.632, -0.054])
#     recenter_list[13]= np.array([0.842, 0.624, -0.053])

#     recenter_list[14]= np.array([0, 0, 0])
recenter_list[14]= np.array([-0.943, -0.179, 0.185])





def deform_arm_only(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    outputs = {
            "xyz": [], "opacities": [], "scales": [], "features_extra": [],
            "rots": [], "features_dc": [], "semantic_id": []
        }
    
    background_id = [1,15]
    active_id = [2,3,4,5,6,7,8,9,10]
    finger_id1 = [11]
    finger_id2 = [12]
    finger_id3 = [13]

    # Your initial & target joint values
    inv_joint_value = torch.tensor(
        [-0.47, 0.07, 0.07, -1.53, 1.5, 1.186, 0.695, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    joint_control_action = torch.tensor(
        [-0.28, -0.205, 0.07, -1.72, 1.426, 0.848, 0.1, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    
    # Number of interpolation steps
    num_subdiv = 2

    for step_idx in range(num_subdiv):
        # fraction in [0..1]
        frac = step_idx / float(num_subdiv - 1)
        # Interpolate the joint values
        current_joint_value = inv_joint_value * (1 - frac) + joint_control_action * frac
        
        # Prepare an outputs dictionary for this frame
        outputs = {
            "xyz": [],
            "opacities": [],
            "scales": [],
            "features_extra": [],
            "rots": [],
            "features_dc": [],
            "semantic_id": []
        }
        
        # -------------------------------------------------------------------
        # 1) For each arm segment in active_id
        # -------------------------------------------------------------------


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
            _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
            inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])
            
            # Kinematic index = mark_id - 2 (based on your snippet)
            kinematic_id = mark_id - 2
            
            inv_rotation_raw = inv_trans[kinematic_id][0:3, 0:3]
            inv_translation  = inv_trans[kinematic_id][0:3, 3]
            
            # Forward transform for current_joint_value
            _, transformation = calculate_franka_mdh_pre_frame(current_joint_value[:12])
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
        
        # -------------------------------------------------------------------
        # 2) For finger_id1, finger_id2, finger_id3
        #    (They share the same kinematic_id=7 in your code)
        # -------------------------------------------------------------------
        for finger_group in [finger_id1, finger_id2, finger_id3]:
            for index in range(len(finger_group)):
                (select_xyz,
                select_opacities, 
                select_scales, 
                select_features_extra,
                select_rotation, 
                select_feature_dc, 
                semantic_id_ind_sam) = filter_with_semantic(
                    semantic_id,
                    finger_group,
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
                
                center_vector_gt  = np.zeros((3,))
                raw_xyz_centered  = torch.from_numpy(select_xyz + center_vector_gt).float()
                
                kinematic_id = 7
                _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
                inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])
                
                inv_rotation_raw = inv_trans[kinematic_id][0:3, 0:3]
                inv_translation  = inv_trans[kinematic_id][0:3, 3]
                
                _, transformation = calculate_franka_mdh_pre_frame(current_joint_value[:12])
                rotation_raw      = transformation[kinematic_id][0:3, 0:3]
                translation       = transformation[kinematic_id][0:3, 3]
                
                deform_point = raw_xyz_centered @ inv_rotation_raw.T + inv_translation
                forward_point = deform_point @ rotation_raw.T + translation
                select_xyz = forward_point
                
                rotation_splat = rotation_raw @ inv_rotation_raw
                
                rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
                rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
                select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)
                
                select_features_extra_deform = sh_rotation_torch(
                    torch.tensor(select_features_extra, dtype=torch.float),
                    select_feature_dc,
                    rotation_splat
                )
                
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
        
        # -------------------------------------------------------------------
        # 3) Collect the output data and save a PLY for this sub-step
        # -------------------------------------------------------------------
        xyz_out       = torch.cat(outputs["xyz"], dim=0).cpu().numpy()
        opacities_out = torch.cat(outputs["opacities"], dim=0).cpu().numpy()
        scales_out    = torch.cat(outputs["scales"], dim=0).cpu().numpy()
        fextra_out    = torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
        rots_out      = torch.cat(outputs["rots"], dim=0).cpu().numpy()
        fdc_out       = torch.cat(outputs["features_dc"], dim=0).transpose(1,2).cpu().numpy()
        sem_out       = torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()
    
        yield xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out
    



def deform_arm(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    outputs = {
            "xyz": [], "opacities": [], "scales": [], "features_extra": [],
            "rots": [], "features_dc": [], "semantic_id": []
        }
    
    background_id = [0,1,14,15]
    active_id = [2,3,4,5,6,7,8,9,10]
    finger_id1 = [11]
    finger_id2 = [12]
    finger_id3 = [13]

    # Your initial & target joint values
    inv_joint_value = torch.tensor(
        [-0.47, 0.07, 0.07, -1.53, 1.5, 1.186, 0.695, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    joint_control_action = torch.tensor(
        [-0.28, -0.205, 0.07, -1.72, 1.426, 0.848, 0.1, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    
    # Number of interpolation steps
    num_subdiv = 200

    for step_idx in range(num_subdiv):
        # fraction in [0..1]
        frac = step_idx / float(num_subdiv - 1)
        # Interpolate the joint values
        current_joint_value = inv_joint_value * (1 - frac) + joint_control_action * frac
        
        # Prepare an outputs dictionary for this frame
        outputs = {
            "xyz": [],
            "opacities": [],
            "scales": [],
            "features_extra": [],
            "rots": [],
            "features_dc": [],
            "semantic_id": []
        }
        
        # -------------------------------------------------------------------
        # 1) For each arm segment in active_id
        # -------------------------------------------------------------------


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
            _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
            inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])
            
            # Kinematic index = mark_id - 2 (based on your snippet)
            kinematic_id = mark_id - 2
            
            inv_rotation_raw = inv_trans[kinematic_id][0:3, 0:3]
            inv_translation  = inv_trans[kinematic_id][0:3, 3]
            
            # Forward transform for current_joint_value
            _, transformation = calculate_franka_mdh_pre_frame(current_joint_value[:12])
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
        
        # -------------------------------------------------------------------
        # 2) For finger_id1, finger_id2, finger_id3
        #    (They share the same kinematic_id=7 in your code)
        # -------------------------------------------------------------------
        for finger_group in [finger_id1, finger_id2, finger_id3]:
            for index in range(len(finger_group)):
                (select_xyz,
                select_opacities, 
                select_scales, 
                select_features_extra,
                select_rotation, 
                select_feature_dc, 
                semantic_id_ind_sam) = filter_with_semantic(
                    semantic_id,
                    finger_group,
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
                
                center_vector_gt  = np.zeros((3,))
                raw_xyz_centered  = torch.from_numpy(select_xyz + center_vector_gt).float()
                
                kinematic_id = 7
                _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
                inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])
                
                inv_rotation_raw = inv_trans[kinematic_id][0:3, 0:3]
                inv_translation  = inv_trans[kinematic_id][0:3, 3]
                
                _, transformation = calculate_franka_mdh_pre_frame(current_joint_value[:12])
                rotation_raw      = transformation[kinematic_id][0:3, 0:3]
                translation       = transformation[kinematic_id][0:3, 3]
                
                deform_point = raw_xyz_centered @ inv_rotation_raw.T + inv_translation
                forward_point = deform_point @ rotation_raw.T + translation
                select_xyz = forward_point
                
                rotation_splat = rotation_raw @ inv_rotation_raw
                
                rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
                rot_matrix_2_transform = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
                select_rotation_deform = matrix_to_quaternion(rot_matrix_2_transform)
                
                select_features_extra_deform = sh_rotation_torch(
                    torch.tensor(select_features_extra, dtype=torch.float),
                    select_feature_dc,
                    rotation_splat
                )
                
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
        
        # -------------------------------------------------------------------
        # 3) Collect the output data and save a PLY for this sub-step
        # -------------------------------------------------------------------
        xyz_out       = torch.cat(outputs["xyz"], dim=0).cpu().numpy()
        opacities_out = torch.cat(outputs["opacities"], dim=0).cpu().numpy()
        scales_out    = torch.cat(outputs["scales"], dim=0).cpu().numpy()
        fextra_out    = torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
        rots_out      = torch.cat(outputs["rots"], dim=0).cpu().numpy()
        fdc_out       = torch.cat(outputs["features_dc"], dim=0).transpose(1,2).cpu().numpy()
        sem_out       = torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()
    
        yield xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out
    



def deform_object(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    background_fixed = [0, 1, 14]
    moving_object_id = [15]
    fixed_arm_and_fingers = [2,3,4,5,6,7,8,9,10,11,12,13]

    angles_deg = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85]
    translations_list = [
        [0.0, -0.0000, -0.0000], [0.0, -0.0000, -0.0200], [0.0, -0.0020, -0.0400],
        [0.0, -0.0035, -0.0600], [0.0, -0.0062, -0.0800], [0.0, -0.0200, -0.1000],
        [0.0, -0.0370, -0.1150], [0.0, -0.0550, -0.1250], [0.0, -0.0900, -0.1350],
        [0.0, -0.1050, -0.1400], [0.0, -0.1150, -0.1420]
    ]

    inv_joint_value = torch.tensor(
        [-0.47, 0.07, 0.07, -1.53, 1.5, 1.186, 0.695, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    final_joint_value = torch.tensor(
        [-0.28, -0.205, 0.07, -1.72, 1.426, 0.848, 0.2, 0, 0, 0, 0, 0],
        dtype=torch.float
    )
    
    num_keyframes = len(angles_deg)
    num_subdiv = 20

    _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
    inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])

    _, transformation = calculate_franka_mdh_pre_frame(final_joint_value[:12])

    for kf in range(num_keyframes - 1):
        angle_start = angles_deg[kf]
        angle_end = angles_deg[kf + 1]
        t_start = torch.tensor(translations_list[kf], dtype=torch.float)
        t_end = torch.tensor(translations_list[kf + 1], dtype=torch.float)

        for substep in range(num_subdiv):
            frac = substep / (num_subdiv - 1)
            angle_now = angle_start * (1 - frac) + angle_end * frac
            t_now = t_start * (1 - frac) + t_end * frac

            q_rad = torch.tensor([angle_now]).deg2rad()

            rotation_splat = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(q_rad), torch.sin(q_rad)],
                [0, -torch.sin(q_rad), torch.cos(q_rad)]
            ], dtype=torch.float).squeeze(0)

            outputs = {k: [] for k in ["xyz", "opacities", "scales",
                                       "features_extra", "rots", "features_dc", "semantic_id"]}

            # Deform arm and fingers based on joint angles
            for arm_id in fixed_arm_and_fingers:
                select_xyz, select_opacities, select_scales, select_features_extra, select_rotation, select_feature_dc, semantic_id_ind_sam = filter_with_semantic(
                    semantic_id, [arm_id], xyz, opacities, scales,
                    features_extra, rots, features_dc, 0
                )

                select_xyz = torch.tensor(select_xyz, dtype=torch.float)
                select_rotation = torch.tensor(select_rotation, dtype=torch.float)
                kinematic_id = arm_id - 2 if arm_id < 11 else 7

                inv_rotation_raw = inv_trans[kinematic_id][:3, :3]
                inv_translation = inv_trans[kinematic_id][:3, 3]

                rotation_raw = transformation[kinematic_id][:3, :3]
                translation = transformation[kinematic_id][:3, 3]

                deform_point = select_xyz @ inv_rotation_raw.T + inv_translation
                forward_point = deform_point @ rotation_raw.T + translation

                rotation_splat_arm = rotation_raw @ inv_rotation_raw
                rot_mat_in = quaternion_to_matrix(select_rotation)
                rot_matrix_transformed = rotation_splat_arm[None, :, :] @ rot_mat_in
                select_rotation_deformed = matrix_to_quaternion(rot_matrix_transformed)

                select_tensors = [forward_point,
                                  torch.tensor(select_opacities, dtype=torch.float),
                                  torch.tensor(select_scales, dtype=torch.float),
                                  torch.tensor(select_features_extra, dtype=torch.float),
                                  select_rotation_deformed,
                                  torch.tensor(select_feature_dc, dtype=torch.float),
                                  torch.tensor(semantic_id_ind_sam, dtype=torch.float)]

                _append_outputs(outputs, *select_tensors)


            # Fixed background, arm, and fingers
            for fixed_id in background_fixed :
                select_data = filter_with_semantic(
                    semantic_id, [fixed_id], xyz, opacities, scales,
                    features_extra, rots, features_dc, 0
                )
                select_tensors = [torch.tensor(sd, dtype=torch.float) for sd in select_data]
                _append_outputs(outputs, *select_tensors)



            # Moving object (id=15)
            select_xyz, select_opacities, select_scales, select_features_extra, select_rotation, select_feature_dc, semantic_id_ind_sam = filter_with_semantic(
                semantic_id, moving_object_id, xyz, opacities, scales,
                features_extra, rots, features_dc, 0
            )

            recenter_vector = recenter_list[14]
            select_xyz_recentered = torch.tensor(select_xyz + recenter_vector, dtype=torch.float)
            select_xyz_deformed = select_xyz_recentered @ rotation_splat.T - recenter_vector + t_now

            rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
            rot_matrix_transformed = torch.matmul(rotation_splat[None, :, :], rot_mat_in)
            select_rotation_deformed = matrix_to_quaternion(rot_matrix_transformed)

            select_features_extra_deformed = sh_rotation_torch(
                torch.tensor(select_features_extra, dtype=torch.float),
                torch.tensor(select_feature_dc, dtype=torch.float),
                rotation_splat
            )

            _append_outputs(outputs,
                            select_xyz_deformed,
                            torch.tensor(select_opacities, dtype=torch.float),
                            torch.tensor(select_scales, dtype=torch.float),
                            select_features_extra_deformed,
                            select_rotation_deformed,
                            torch.tensor(select_feature_dc, dtype=torch.float),
                            torch.tensor(semantic_id_ind_sam, dtype=torch.float))

            xyz_out = torch.cat(outputs["xyz"], dim=0).cpu().numpy()
            opacities_out = torch.cat(outputs["opacities"], dim=0).cpu().numpy()
            scales_out = torch.cat(outputs["scales"], dim=0).cpu().numpy()
            fextra_out = torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
            rots_out = torch.cat(outputs["rots"], dim=0).cpu().numpy()
            fdc_out = torch.cat(outputs["features_dc"], dim=0).transpose(1, 2).cpu().numpy()
            sem_out = torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()

            yield xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out


def deform_finger(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    background_fixed = [0, 1, 14, 15]
    fixed_arm_and_fingers = [2,3,4,5,6,7,8,9,10]
    finger_ids = [[11], [12], [13]]

    inv_joint_value = torch.tensor([-0.47, 0.07, 0.07, -1.53, 1.5, 1.186, 0.695, 0, 0, 0, 0, 0], dtype=torch.float)
    joint_control_action = torch.tensor([-0.28, -0.205, 0.07, -1.72, 1.426, 0.848, 0.1, 0, 0, 0, 0, 0], dtype=torch.float)

    num_steps = 70
    _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
    inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])

    _, fwd_transformation = calculate_franka_mdh_pre_frame(joint_control_action[:12])

    for step_idx in range(num_steps):
        frac = step_idx / (num_steps - 1)

        outputs = {k: [] for k in ["xyz", "opacities", "scales", "features_extra", "rots", "features_dc", "semantic_id"]}

        for fixed_id in background_fixed:
            select_data = filter_with_semantic(semantic_id, [fixed_id], xyz, opacities, scales, features_extra, rots, features_dc, 0)
            select_tensors = [torch.tensor(sd, dtype=torch.float) for sd in select_data]
            _append_outputs(outputs, *select_tensors)

        for arm_id in fixed_arm_and_fingers:
            select_xyz, select_opacities, select_scales, select_features_extra, select_rotation, select_feature_dc, semantic_id_ind_sam = filter_with_semantic(
                semantic_id, [arm_id], xyz, opacities, scales, features_extra, rots, features_dc, 0
            )
            select_xyz = torch.tensor(select_xyz, dtype=torch.float)
            # select_rotation = torch.tensor(select_rotation, dtype=torch.float)
            kinematic_id = arm_id - 2

            deform_point = select_xyz @ inv_trans[kinematic_id][:3, :3].T + inv_trans[kinematic_id][:3, 3]
            forward_point = deform_point @ fwd_transformation[kinematic_id][:3, :3].T + fwd_transformation[kinematic_id][:3, 3]

            rotation_splat_arm = fwd_transformation[kinematic_id][:3, :3] @ inv_trans[kinematic_id][:3, :3]

            rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
            rot_matrix_transformed = rotation_splat_arm @ rot_mat_in
            select_rotation_deformed = matrix_to_quaternion(rot_matrix_transformed)

            select_features_extra_deformed = sh_rotation_torch(
                torch.tensor(select_features_extra, dtype=torch.float),
                torch.tensor(select_feature_dc, dtype=torch.float),
                rotation_splat_arm
            )

            _append_outputs(outputs,
                            torch.tensor(forward_point, dtype=torch.float),
                            torch.tensor(select_opacities, dtype=torch.float),
                            torch.tensor(select_scales, dtype=torch.float),
                            select_features_extra_deformed,
                            select_rotation_deformed,
                            torch.tensor(select_feature_dc, dtype=torch.float),
                            torch.tensor(semantic_id_ind_sam, dtype=torch.float))
        # Finger deformation specifics
        angle_start_deg, angle_end_deg = 10.0, -60.0

        # Constant 15-degree rotation offset around the Y-axis
        z_offset_rad = torch.deg2rad(torch.tensor(-7.0))
        # Rotation around Z-axis
        z_offset_rot = torch.tensor([
            [torch.cos(z_offset_rad), -torch.sin(z_offset_rad), 0],
            [torch.sin(z_offset_rad), torch.cos(z_offset_rad), 0],
            [0, 0, 1]
        ])




        recenter_vecs = [
            torch.tensor([-0.945, -0.613, -0.05]),
            torch.tensor([-0.926, -0.609, -0.06]),
            torch.tensor([-0.852, -0.616, -0.06])
        ]

        trans_starts = [
            torch.tensor([0.065, -0.374, -0.07]),
            torch.tensor([0.065, -0.364, -0.07]),
            torch.tensor([0.065, -0.36, -0.07])
        ]


        trans_ends = [
            torch.tensor([0.065, -0.334, -0.14]),
            torch.tensor([0.065, -0.334, -0.14]),
            torch.tensor([0.065, -0.334, -0.14])
        ]
        current_angle_deg = angle_start_deg * (1 - frac) + angle_end_deg * frac
        current_angle_rad = torch.deg2rad(torch.tensor(current_angle_deg))

        finger_rot = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(current_angle_rad), torch.sin(current_angle_rad)],
            [0, -torch.sin(current_angle_rad), torch.cos(current_angle_rad)]
        ])

        finger_rot = finger_rot @ z_offset_rot

        for finger_id, recenter, trans_start, trans_end in zip(finger_ids, recenter_vecs, trans_starts, trans_ends):
            finger_offset = (1 - frac) * trans_start + frac * trans_end
            select_data = filter_with_semantic(semantic_id, finger_id, xyz, opacities, scales, features_extra, rots, features_dc, 0)
            fx_xyz, fx_opac, fx_scales, fx_fextra, fx_rot, fx_fdc, fx_sem_id = [torch.tensor(d, dtype=torch.float) for d in select_data]

            xyz_recentered = fx_xyz + recenter
            xyz_deformed = xyz_recentered @ finger_rot.T - recenter + finger_offset

            rot_mat_in = quaternion_to_matrix(fx_rot)
            rot_matrix_transformed = finger_rot @ rot_mat_in
            fx_rot_deformed = matrix_to_quaternion(rot_matrix_transformed)

            fx_fextra_deformed = sh_rotation_torch(fx_fextra, fx_fdc, finger_rot)

            _append_outputs(outputs, xyz_deformed, fx_opac, fx_scales, fx_fextra_deformed, fx_rot_deformed, fx_fdc, fx_sem_id)

        xyz_out = torch.cat(outputs["xyz"], dim=0).cpu().numpy()
        opacities_out = torch.cat(outputs["opacities"], dim=0).cpu().numpy()
        scales_out = torch.cat(outputs["scales"], dim=0).cpu().numpy()
        fextra_out = torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
        rots_out = torch.cat(outputs["rots"], dim=0).cpu().numpy()
        fdc_out = torch.cat(outputs["features_dc"], dim=0).transpose(1, 2).cpu().numpy()
        sem_out = torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()

        yield xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out




def deform_finger_only(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    background_fixed = [1, 15]
    fixed_arm_and_fingers = [2,3,4,5,6,7,8,9,10]
    finger_ids = [[11], [12], [13]]

    inv_joint_value = torch.tensor([-0.47, 0.07, 0.07, -1.53, 1.5, 1.186, 0.695, 0, 0, 0, 0, 0], dtype=torch.float)
    joint_control_action = torch.tensor([-0.28, -0.205, 0.07, -1.72, 1.426, 0.848, 0.1, 0, 0, 0, 0, 0], dtype=torch.float)

    num_steps = 2
    _, inv_transformation = calculate_franka_mdh_pre_frame(inv_joint_value[:12])
    inv_trans = inverse_affine_transformation_torch(inv_transformation[:12])

    _, fwd_transformation = calculate_franka_mdh_pre_frame(joint_control_action[:12])

    for step_idx in range(num_steps):
        frac = step_idx / (num_steps - 1)

        outputs = {k: [] for k in ["xyz", "opacities", "scales", "features_extra", "rots", "features_dc", "semantic_id"]}

        for fixed_id in background_fixed:
            select_data = filter_with_semantic(semantic_id, [fixed_id], xyz, opacities, scales, features_extra, rots, features_dc, 0)
            select_tensors = [torch.tensor(sd, dtype=torch.float) for sd in select_data]
            _append_outputs(outputs, *select_tensors)

        for arm_id in fixed_arm_and_fingers:
            select_xyz, select_opacities, select_scales, select_features_extra, select_rotation, select_feature_dc, semantic_id_ind_sam = filter_with_semantic(
                semantic_id, [arm_id], xyz, opacities, scales, features_extra, rots, features_dc, 0
            )
            select_xyz = torch.tensor(select_xyz, dtype=torch.float)
            # select_rotation = torch.tensor(select_rotation, dtype=torch.float)
            kinematic_id = arm_id - 2

            deform_point = select_xyz @ inv_trans[kinematic_id][:3, :3].T + inv_trans[kinematic_id][:3, 3]
            forward_point = deform_point @ fwd_transformation[kinematic_id][:3, :3].T + fwd_transformation[kinematic_id][:3, 3]

            rotation_splat_arm = fwd_transformation[kinematic_id][:3, :3] @ inv_trans[kinematic_id][:3, :3]

            rot_mat_in = quaternion_to_matrix(torch.tensor(select_rotation, dtype=torch.float))
            rot_matrix_transformed = rotation_splat_arm @ rot_mat_in
            select_rotation_deformed = matrix_to_quaternion(rot_matrix_transformed)

            select_features_extra_deformed = sh_rotation_torch(
                torch.tensor(select_features_extra, dtype=torch.float),
                torch.tensor(select_feature_dc, dtype=torch.float),
                rotation_splat_arm
            )

            _append_outputs(outputs,
                            torch.tensor(forward_point, dtype=torch.float),
                            torch.tensor(select_opacities, dtype=torch.float),
                            torch.tensor(select_scales, dtype=torch.float),
                            select_features_extra_deformed,
                            select_rotation_deformed,
                            torch.tensor(select_feature_dc, dtype=torch.float),
                            torch.tensor(semantic_id_ind_sam, dtype=torch.float))
        # Finger deformation specifics
        angle_start_deg, angle_end_deg = 10.0, -60.0

        # Constant 15-degree rotation offset around the Y-axis
        z_offset_rad = torch.deg2rad(torch.tensor(-7.0))
        # Rotation around Z-axis
        z_offset_rot = torch.tensor([
            [torch.cos(z_offset_rad), -torch.sin(z_offset_rad), 0],
            [torch.sin(z_offset_rad), torch.cos(z_offset_rad), 0],
            [0, 0, 1]
        ])


        recenter_vecs = [
            torch.tensor([-0.945, -0.613, -0.05]),
            torch.tensor([-0.926, -0.609, -0.06]),
            torch.tensor([-0.852, -0.616, -0.06])
        ]

        trans_starts = [
            torch.tensor([0.065, -0.374, -0.07]),
            torch.tensor([0.065, -0.364, -0.07]),
            torch.tensor([0.065, -0.36, -0.07])
        ]

        trans_ends = [
            torch.tensor([0.065, -0.334, -0.14]),
            torch.tensor([0.065, -0.334, -0.14]),
            torch.tensor([0.065, -0.334, -0.14])
        ]

        current_angle_deg = angle_start_deg * (1 - frac) + angle_end_deg * frac
        current_angle_rad = torch.deg2rad(torch.tensor(current_angle_deg))

        finger_rot = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(current_angle_rad), torch.sin(current_angle_rad)],
            [0, -torch.sin(current_angle_rad), torch.cos(current_angle_rad)]
        ])

        finger_rot = finger_rot @ z_offset_rot

        for finger_id, recenter, trans_start, trans_end in zip(finger_ids, recenter_vecs, trans_starts, trans_ends):
            finger_offset = (1 - frac) * trans_start + frac * trans_end
            select_data = filter_with_semantic(semantic_id, finger_id, xyz, opacities, scales, features_extra, rots, features_dc, 0)
            fx_xyz, fx_opac, fx_scales, fx_fextra, fx_rot, fx_fdc, fx_sem_id = [torch.tensor(d, dtype=torch.float) for d in select_data]

            xyz_recentered = fx_xyz + recenter
            xyz_deformed = xyz_recentered @ finger_rot.T - recenter + finger_offset

            rot_mat_in = quaternion_to_matrix(fx_rot)
            rot_matrix_transformed = finger_rot @ rot_mat_in
            fx_rot_deformed = matrix_to_quaternion(rot_matrix_transformed)

            fx_fextra_deformed = sh_rotation_torch(fx_fextra, fx_fdc, finger_rot)

            _append_outputs(outputs, xyz_deformed, fx_opac, fx_scales, fx_fextra_deformed, fx_rot_deformed, fx_fdc, fx_sem_id)

        xyz_out = torch.cat(outputs["xyz"], dim=0).cpu().numpy()
        opacities_out = torch.cat(outputs["opacities"], dim=0).cpu().numpy()
        scales_out = torch.cat(outputs["scales"], dim=0).cpu().numpy()
        fextra_out = torch.cat(outputs["features_extra"], dim=0).cpu().numpy()
        rots_out = torch.cat(outputs["rots"], dim=0).cpu().numpy()
        fdc_out = torch.cat(outputs["features_dc"], dim=0).transpose(1, 2).cpu().numpy()
        sem_out = torch.cat(outputs["semantic_id"], dim=0).cpu().numpy()

        yield xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out


        
def deform_scene_combined(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id):
    finger_steps = list(deform_finger(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id))
    object_steps = list(deform_object(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id))

    finger_ids = [11, 12, 13]
    object_id = 15

    # Frames 0–22: finger deformation only
    for step in range(50):
        yield finger_steps[step]

    base_finger_frame = finger_steps[50]
    non_object_mask = ~(np.isin(base_finger_frame[6], finger_ids + [object_id])).reshape(-1)

    for finger_step, object_step in zip(finger_steps[50:70], object_steps[0:20]):
        finger_mask = np.isin(finger_step[6], finger_ids).reshape(-1)
        object_mask = (object_step[6] == object_id).reshape(-1)

        xyz_merged = np.concatenate([
            base_finger_frame[0][non_object_mask],
            finger_step[0][finger_mask],
            object_step[0][object_mask]
        ], axis=0)

        opacities_merged = np.concatenate([
            base_finger_frame[1][non_object_mask],
            finger_step[1][finger_mask],
            object_step[1][object_mask]
        ], axis=0)

        scales_merged = np.concatenate([
            base_finger_frame[2][non_object_mask],
            finger_step[2][finger_mask],
            object_step[2][object_mask]
        ], axis=0)

        fextra_merged = np.concatenate([
            base_finger_frame[3][non_object_mask],
            finger_step[3][finger_mask],
            object_step[3][object_mask]
        ], axis=0)

        rots_merged = np.concatenate([
            base_finger_frame[4][non_object_mask],
            finger_step[4][finger_mask],
            object_step[4][object_mask]
        ], axis=0)

        fdc_merged = np.concatenate([
            base_finger_frame[5][non_object_mask],
            finger_step[5][finger_mask],
            object_step[5][object_mask]
        ], axis=0)

        sem_merged = np.concatenate([
            base_finger_frame[6][non_object_mask],
            finger_step[6][finger_mask],
            object_step[6][object_mask]
        ], axis=0)

        yield (xyz_merged, opacities_merged, scales_merged,
               fextra_merged, rots_merged, fdc_merged, sem_merged)
    # Frames 31–100: object deformation with fingers fixed at last frame of 30
    base_finger_frame_30 = xyz_merged, opacities_merged, scales_merged, fextra_merged, rots_merged, fdc_merged, sem_merged
    non_object_mask_final = (base_finger_frame_30[6] != object_id).reshape(-1)

    for step in range(20, len(object_steps)):
        object_step = object_steps[step]
        object_mask = (object_step[6] == object_id).reshape(-1)

        xyz_merged_final = np.concatenate([
            base_finger_frame_30[0][non_object_mask_final],
            object_step[0][object_mask]
        ], axis=0)

        opacities_merged_final = np.concatenate([
            base_finger_frame_30[1][non_object_mask_final],
            object_step[1][object_mask]
        ], axis=0)

        scales_merged_final = np.concatenate([
            base_finger_frame_30[2][non_object_mask_final],
            object_step[2][object_mask]
        ], axis=0)

        fextra_merged_final = np.concatenate([
            base_finger_frame_30[3][non_object_mask_final],
            object_step[3][object_mask]
        ], axis=0)

        rots_merged_final = np.concatenate([
            base_finger_frame_30[4][non_object_mask_final],
            object_step[4][object_mask]
        ], axis=0)

        fdc_merged_final = np.concatenate([
            base_finger_frame_30[5][non_object_mask_final],
            object_step[5][object_mask]
        ], axis=0)

        sem_merged_final = np.concatenate([
            base_finger_frame_30[6][non_object_mask_final],
            object_step[6][object_mask]
        ], axis=0)

        yield (xyz_merged_final, opacities_merged_final, scales_merged_final,
               fextra_merged_final, rots_merged_final, fdc_merged_final, sem_merged_final)