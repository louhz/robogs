

import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree as KDTree

# ---------------------------------------------------------------------------------
# Reuse your given helper functions for loading/saving .ply
# (You can copy them verbatim, only including here for completeness)
# ---------------------------------------------------------------------------------


def construct_list_of_attributes():
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(3):
        l.append(f"f_dc_{i}")
    for i in range(45):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(3):
        l.append(f"scale_{i}")
    for i in range(4):
        l.append(f"rot_{i}")
    return l


def construct_list_of_attributes_sam():
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        l.append(f"f_dc_{i}")
    for i in range(45):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    l.append("semantic_id")
    for i in range(3):
        l.append(f"scale_{i}")
    for i in range(4):
        l.append(f"rot_{i}")
    return l


def load_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]), np.array(plydata.elements[0]["y"]), np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 1, 3), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 0, 1] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 0, 2] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    # e.g. 3 * (max_sh_degree + 1) ^ 2 - 3  ->  3*(3+1)^2 - 3 = 3*16 - 3 = 48 - 3 = 45
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    # reshape to (num_points, 3, (#SHcoeffs except DC))
    
    features_extra = features_extra.reshape((features_extra.shape[0], 3, 15))

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


def load_ply_sam(path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (np.array(plydata.elements[0]["x"]), np.array(plydata.elements[0]["y"]), np.array(plydata.elements[0]["z"])),
        axis=1,
    )
    opacities = np.array(plydata.elements[0]["opacity"])[..., np.newaxis]
    semantic_id = np.array(plydata.elements[0]["semantic_id"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.array(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.array(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.array(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.array(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.array(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots, semantic_id


def save_ply(xyz, f_dc, f_rest, opacities, scale, rotation, path):
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # Flatten data into a single array of shape (N, total_channels)
    attributes = np.concatenate(
        (
            xyz,
            normals,
            f_dc.reshape((xyz.shape[0], -1)),
            f_rest.reshape((xyz.shape[0], -1)),
            opacities,
            scale,
            rotation,
        ),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def save_ply_sam(xyz, f_dc, f_rest, opacities, semantic_id, scale, rotation, path):
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes_sam()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # Flatten data into a single array of shape (N, total_channels)
    attributes = np.concatenate(
        (
            xyz,
            normals,
            f_dc.reshape((xyz.shape[0], -1)),
            f_rest.reshape((xyz.shape[0], -1)),
            opacities,
            semantic_id,
            scale,
            rotation,
        ),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


# ---------------------------------------------------------------------------------
# Main code to assign semantic IDs and save the updated scene
# ---------------------------------------------------------------------------------


def assign_semantic_ids(base_ply_path, part_info_list, output_ply_path, max_sh_degree=3, search_radius=1e-6):
    """
    base_ply_path: Path to the original scene .ply WITHOUT semantic IDs (unassigned).
    part_info_list: List of tuples: [ (part_ply_path_or_xyz, semantic_id), ... ]
        where `part_ply_path_or_xyz` can be either:
          - A string path to a .ply that we can load (with load_ply, or load_ply_sam).
          - A numpy array of shape (N, 3) specifying the xyz positions of that part's points.
        and `semantic_id` is an integer specifying the ID for that part.
    output_ply_path: Path to save the final .ply that includes the semantic_id channel.
    max_sh_degree: The maximum spherical harmonics degree in your scene.
    search_radius: Tolerance for “nearest” points. If your geometry is perfect,
                   you might set this very small (1e-8).
                   If there’s noise, you might need something larger.
    """

    # 1) Load the BASE mesh (no semantic ID in it)
    base_xyz, base_dc, base_rest, base_opacity, base_scale, base_rot = load_ply(base_ply_path, max_sh_degree)
    N = base_xyz.shape[0]

    # 2) Initialize a semantic_id array to zeros
    semantic_id = np.zeros((N, 1), dtype=np.float32)

    # 3) Build a KD-tree on the base geometry
    tree = KDTree(base_xyz)

    # 4) For each part, assign its semantic ID to the matching base points
    for part_data, sid in part_info_list:
        # (A) If part_data is a path to a .ply, load it.
        if isinstance(part_data, str):
            # It might be a standard ply or one with semantic_id.
            # For safety, use 'load_ply' if you only need xyz.
            p_xyz, p_dc, p_rest, p_opacity, p_scale, p_rot = load_ply(part_data, max_sh_degree)
        else:
            # (B) If part_data is already an (N,3) numpy array
            p_xyz = part_data

        # (C) For each point in p_xyz, find the nearest neighbor in base_xyz
        #     If it’s within search_radius, assign that base point the semantic ID = sid
        distances, indices = tree.query(p_xyz, k=1)

        # (D) Filter out indices that are within the threshold distance
        close_mask = distances <= search_radius
        valid_indices = indices[close_mask]

        semantic_id[valid_indices] = sid

    # 5) Save the new PLY with the semantic_id channel
    save_ply_sam(
        xyz=base_xyz,
        f_dc=base_dc,
        f_rest=base_rest,
        opacities=base_opacity,
        semantic_id=semantic_id,
        scale=base_scale,
        rotation=base_rot,
        path=output_ply_path,
    )


def filter_ply_with_id(ply_path, semantic_class_id):
    """
    Filters the PLY data based on a given semantic class ID.

    Args:
        ply_path (str): Path to the PLY file.
        semantic_class_id (int): The ID to filter points by.

    Returns:
        tuple: Filtered (xyz, features_dc, features_extra, opacities, scales, rots).
    """
    # Load PLY data
    xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = load_ply_sam(ply_path)

    # Find valid indices where semantic_id matches the given class ID
    valid_indices = np.where(semantic_id == semantic_class_id)[0]  # Ensure it's an array of indices

    # Apply filtering
    return (
        xyz[valid_indices],
        features_dc[valid_indices],
        features_extra[valid_indices],
        opacities[valid_indices],
        scales[valid_indices],
        rots[valid_indices],
        semantic_id[valid_indices],
    )


# if __name__ == "__main__":

#     # Example usage:
#     base_scene = (  # your original scene with no semantic ID
#         "/home/haozhe/Dropbox/rendering/asset/scene.ply"
#     )
#     out_scene = "/home/haozhe/Dropbox/rendering/asset/final_scene_with_ids.ply"
#     part_scene = "/home/haozhe/Dropbox/rendering/asset/part"

#     if not os.path.exists(part_scene):
#         os.makedirs(part_scene)
#     arm_path = "/home/haozhe/Dropbox/rendering/asset/arm"
#     hand_path = "/home/haozhe/Dropbox/rendering/asset/hand"
#     table_path = "/home/haozhe/Dropbox/rendering/asset/table"
#     object_path = "/home/haozhe/Dropbox/rendering/asset/object"


#     # this is the key dictonary to assign the semantic id 
#     # you need to assign this part
#     parts = [
#         (os.path.join(arm_path, "link0.ply"), 1),
#         (os.path.join(arm_path, "link1.ply"), 2),
#         (os.path.join(arm_path, "link2.ply"), 3),
#         (os.path.join(arm_path, "link3.ply"), 4),
#         (os.path.join(arm_path, "link4.ply"), 5),
#         (os.path.join(arm_path, "link5.ply"), 6),
#         (os.path.join(arm_path, "link6.ply"), 7),
#         (os.path.join(arm_path, "link7.ply"), 8),
#         (os.path.join(hand_path, "hand.ply"), 9),
#         (os.path.join(hand_path, "finger1.ply"), 10),
#         (os.path.join(hand_path, "finger2.ply"), 11),
#         (os.path.join(hand_path, "finger3.ply"), 12),
#         (os.path.join(hand_path, "finger4.ply"), 13),
#         (os.path.join(table_path, "tablemesh.ply"), 14),
#         (os.path.join(object_path, "object.ply"), 15),
#     ]

#     assign_semantic_ids(
#         base_ply_path=base_scene, part_info_list=parts, output_ply_path=out_scene, max_sh_degree=3, search_radius=1e-6
#     )

#     class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#     for class_id in class_ids:
#         filtered_data = filter_ply_with_id(out_scene, class_id)
#         (
#             filtered_xyz,
#             filtered_features_dc,
#             filtered_features_extra,
#             filtered_opacities,
#             filtered_scales,
#             filtered_rots,
#             semantic_id,
#         ) = filtered_data

#         # Save the filtered data to a new PLY file
#         output_path = os.path.join(part_scene, f"filtered_class_{class_id}.ply")

#         save_ply_sam(
#             filtered_xyz,
#             filtered_features_dc,
#             filtered_features_extra,
#             filtered_opacities,
#             semantic_id,
#             filtered_scales,
#             filtered_rots,
#             output_path,
#         )
#         print(f"Filtered data for class {class_id} saved to: {output_path}")
#     print(f"Done! Saved new PLY with semantic IDs to: {out_scene}")




if __name__ == "__main__":

    # Example usage:
    base_scene = (  # your original scene with no semantic ID
        "/home/haozhe/Dropbox/ucb/ucb/asset/scene.ply"
    )
    out_scene = "/home/haozhe/Dropbox/ucb/ucb/asset/final_scene_with_ids.ply"
    part_scene = "/home/haozhe/Dropbox/ucb/ucb/asset/part"

    if not os.path.exists(part_scene):
        os.makedirs(part_scene)
    arm_path = "/home/haozhe/Dropbox/ucb/ucb/asset/arm"
    hand_path = "/home/haozhe/Dropbox/ucb/ucb/asset/gripper"
    # table_path = "/home/haozhe/Dropbox/ucb/ucb/asset/table"
    object_path = "/home/haozhe/Dropbox/ucb/ucb/asset/object"



    # this is the key dictonary to assign the semantic id 
    # you need to assign this part
    parts = [
        (os.path.join(arm_path, "link0.ply"), 1),
        (os.path.join(arm_path, "link1.ply"), 2),
        (os.path.join(arm_path, "link2.ply"), 3),
        (os.path.join(arm_path, "link3.ply"), 4),
        (os.path.join(arm_path, "link4.ply"), 5),
        (os.path.join(arm_path, "link5.ply"), 6),
        (os.path.join(arm_path, "link6.ply"), 7),
        (os.path.join(arm_path, "link7.ply"), 8),
        (os.path.join(hand_path, "base.ply"), 9),
        (os.path.join(hand_path, "gripperleft1.ply"), 10),
        (os.path.join(hand_path, "gripperleft2.ply"), 11),
        (os.path.join(hand_path, "gripperright1.ply"), 12),
        (os.path.join(hand_path, "gripperright2.ply"), 13),
        # (os.path.join(table_path, "tablemesh.ply"), 14),
        (os.path.join(object_path, "object.ply"), 14),
    ]

    assign_semantic_ids(
        base_ply_path=base_scene, part_info_list=parts, output_ply_path=out_scene, max_sh_degree=3, search_radius=1e-6
    )

    class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for class_id in class_ids:
        filtered_data = filter_ply_with_id(out_scene, class_id)
        (
            filtered_xyz,
            filtered_features_dc,
            filtered_features_extra,
            filtered_opacities,
            filtered_scales,
            filtered_rots,
            semantic_id,
        ) = filtered_data

        # Save the filtered data to a new PLY file
        output_path = os.path.join(part_scene, f"filtered_class_{class_id}.ply")

        save_ply_sam(
            filtered_xyz,
            filtered_features_dc,
            filtered_features_extra,
            filtered_opacities,
            semantic_id,
            filtered_scales,
            filtered_rots,
            output_path,
        )
        print(f"Filtered data for class {class_id} saved to: {output_path}")
    print(f"Done! Saved new PLY with semantic IDs to: {out_scene}")
