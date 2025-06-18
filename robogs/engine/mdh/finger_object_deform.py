
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
from gsplat.strategy import DefaultStrategy, MCMCStrategy


def render_simulation():

    class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    class_name= [
        (("link0"), 1),
        (("link1"), 2),
        (("link2"), 3),
        (("link3"), 4),
        (("link4"), 5),
        (("link5"), 6),
        (("link6"), 7),
        (("link7"), 8),
        (("palm"), 9),
        (("if_bs"), 10),
        (("mf_bs"), 11),
        (("rf_bs"), 12),
        (("th_ds"), 13),
        (("tablemesh"), 14),
        (("ketchup"), 15),
    ]


    body_names=['attachment', 'ketchup', 'if_bs', 'if_ds', 'if_md', 'if_px',
                 'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                   'mf_bs', 'mf_ds', 'mf_md', 'mf_px',
                     'palm', 'plate', 
                     'rf_bs', 'rf_ds', 'rf_md', 'rf_px', 'th_bs',
                       'th_ds', 'th_mp', 'th_px', 'world']
    mj_file = '/home/haozhe/Dropbox/physics/franka_leap_demo/scene_ketchup_render.xml'
    mj_model = mj.MjModel.from_xml_path(mj_file)
    mj_data = mj.MjData(mj_model)


    # setup object mass:
    mj_data.model.body_mass[mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, 'ketchup')] = 1.5
    mj_data.model.body_mass[mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, 'ketchup')] = 0.7

    robot_action = np.array([-0.47, 0.07, 0.07, -1.6, 1.5, 1.186, 0.695])

    # query position and orientation from mujoco given the robot_action
    # Launch the passive viewer
    with mv.launch_passive(mj_model, mj_data) as viewer:
        # We’ll run 2000 steps or until the viewer is closed.
        for step in range(500):
            # If the user closes the viewer window, stop the loop.
            if not viewer.is_running():
                break

            # Set the control input
            mj_data.ctrl[:7] = robot_action

            # Step the physics simulation
            mujoco.mj_step(mj_model, mj_data)


            # Update the viewer
            viewer.sync()

            # Slow down the loop for visualization
            time.sleep(0.01)

    print("Simulation finished or viewer closed.")
    # find the initial inverse transformation

    qpos_degree=7+16+7 
    # get the position and orientation of the robot
    inv_pos_body =np.zeros((len(body_names), 3))
    inv_quat_body = np.zeros((len(body_names), 3,3))
    inv_joint_value= np.zeros((qpos_degree, 1))
    inv_hand_action = np.zeros((len(body_names), 16))
    for i, body_name in enumerate(body_names):
        body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, body_name)
        inv_pos_body[i] = mj_data.xpos[body_id]
        # inv_pos_body[i] = mj_data.xipos[body_id]
        inv_quat_body[i] = mj_data.ximat[body_id].reshape(3,3)
    
    for i in range(qpos_degree):
        inv_joint_value[i] = mj_data.qpos[i]


    
    # robot_action_deform = np.array([-0.07, 0.3, 0.017, -2.07, 0.017, 2.39,-0.174])
    robot_action_deform = np.array([-0.488, 0.14 ,0.31, -1.95, 1.36 ,1.43, 0.25])
    with mv.launch_passive(mj_model, mj_data) as viewer:
        # We’ll run 2000 steps or until the viewer is closed.
        for step in range(100):
            # If the user closes the viewer window, stop the loop.
            if not viewer.is_running():
                break

            # Set the control input
            mj_data.ctrl[:7] = robot_action_deform

            # Step the physics simulation
            mujoco.mj_step(mj_model, mj_data)


            # Update the viewer
            viewer.sync()

            # Slow down the loop for visualization
            time.sleep(0.01)

    print("Simulation finished or viewer closed.")


    hand_action = np.array([
        0.6,0,0,0,
        0.6,0,0,0,
        0.6,0,0,0,
        0,0,0,0,
    ])


    
 
    inv_pos=np.zeros((len(class_ids), 3))
    inv_quat=np.zeros((len(class_ids), 3,3))
    

    

    # # find quat of the current object and links 

    # # match the class_name and body_names to reassign the inv_pos,inv_quat, pos, quat
    for i, (name, id) in enumerate(class_name):
        # find the index of the body name in the body_names
        for j, body_name in enumerate(body_names):
            if name in body_name:
                # assign the inv_pos and inv_quat to the corresponding index
                inv_pos[id] = inv_pos_body[j]
                inv_quat[id] = inv_quat_body[j]
    #             pos[id] = pos_body[j]
    #             quat[id] = quat_body[j]

    num_steps = 200

    # Launch your viewer
    with mv.launch_passive(mj_model, mj_data) as viewer:
        # Pre-allocate arrays: time x number_of_class_ids x ...
        pos = np.zeros((num_steps, len(class_ids), 3))
        quat = np.zeros((num_steps, len(class_ids), 3, 3))


        # If you need to store the "raw" body data for each step before matching:
        pos_body = np.zeros((num_steps, len(body_names), 3))
        quat_body = np.zeros((num_steps, len(body_names), 3, 3))
        joint_control = np.zeros((num_steps, qpos_degree))

        
        # We’ll run num_steps steps or until the viewer is closed.
        for step in range(num_steps):
            # If the user closes the viewer window, stop the loop.
            if not viewer.is_running():
                break

            # Set the control input (example)
            mj_data.ctrl[7:23] = hand_action

            # Step the physics simulation
            mujoco.mj_step(mj_model, mj_data)

            # Update the viewer
            viewer.sync()

            # Slow down the loop for visualization
            time.sleep(0.01)

            # ----------------------------------------------
            # Collect data for the current step:
            # ----------------------------------------------

            # 1. Fill pos_body and quat_body for each body
            for i, body_name in enumerate(body_names):
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

                # Global position
                pos_body[step, i] = mj_data.xpos[body_id]

                # Global orientation as 3x3 rotation matrix
                quat_body[step, i] = mj_data.ximat[body_id].reshape(3, 3)

                # for k in range(qpos_degree):
                #     joint_control[step, k] = mj_data.qpos[body_id]

            # 2. Match the class_name and body_names to reassign inv_pos, inv_quat, pos, quat
            for (name, cid) in class_name:
                for j, body_name in enumerate(body_names):
                    # If there's a match in the name
                    if name in body_name:

                        pos[step, cid] = pos_body[step, j]
                        quat[step, cid] = quat_body[step, j]

        print("Simulation finished or viewer closed.")

    # if values of quat in inv_pos is (0,0,0,0) then make it (1,0,0,0)
    for i in range(len(inv_quat)):
        if np.all(inv_quat[i] == 0):
            inv_quat[i] = np.array([[1, 0, 0],  
                                    [0, 1, 0],
                                    [0, 0, 1]])
        if np.all(quat[i] == 0):
            inv_quat[i] = np.array([[1, 0, 0],  
                                    [0, 1, 0],
                                    [0, 0, 1]])
    #  or some subset, depending on your use case.
    
    # If you have some “center of gravity” or reference center:


    # print out the trajectory difference and the rotation difference of object


    translations_list = [
        [0.0,  -0.0000, -0.0000],  # t0
        [0.0,  -0.0000, -0.0200],  # t1
        [0.0,  -0.0020, -0.0400],  # t2
        [0.0,  -0.0035, -0.0600],  # t3
        [0.0,  -0.0062, -0.0800],  # t4
        [0.0,  -0.0200, -0.1000],  # t5
        [0.0,  -0.0370, -0.1150],  # t6
        [0.0,  -0.0550, -0.1250],  # t7
        [0.0,  -0.0900, -0.1350],  # t8
        [0.0,  -0.1050, -0.1400],  # t9
        [0.0,  -0.1150, -0.1420],  # t10
    ]

    angles_deg = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85]