import xml.etree.ElementTree as ET

import pypose as pp
import torch


def euler_to_quaternion_torch(roll, pitch, yaw):
    """
    Convert Tait-Bryan Euler angles (roll, pitch, yaw) to quaternions (w, x, y, z)
    using PyTorch operations. All angles are assumed to be in radians, with
    intrinsic rotations about X, then Y, then Z (roll-pitch-yaw).

    Args:
        roll (torch.Tensor): Rotation about X-axis, in radians.
        pitch (torch.Tensor): Rotation about Y-axis, in radians.
        yaw (torch.Tensor): Rotation about Z-axis, in radians.

    Returns:
        torch.Tensor: Quaternions with shape (..., 4). The last dimension is (w, x, y, z).
    """

    # Ensure all angles have the same shape.
    # For single values, you can pass torch.tensor(scalar).

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)


import numpy as np


def euler_to_quaternion_numpy(roll, pitch, yaw):
    """
    Convert Tait-Bryan Euler angles (roll, pitch, yaw) to quaternions (w, x, y, z)
    using NumPy operations. All angles are assumed to be in radians, with
    intrinsic rotations about X, then Y, then Z (roll-pitch-yaw).

    Args:
        roll (float or np.ndarray): Rotation about X-axis, in radians.
        pitch (float or np.ndarray): Rotation about Y-axis, in radians.
        yaw (float or np.ndarray): Rotation about Z-axis, in radians.

    Returns:
        np.ndarray: Quaternions with shape (..., 4). The last dimension is (w, x, y, z).
    """

    # Convert to arrays if scalars are given
    roll = np.asarray(roll)
    pitch = np.asarray(pitch)
    yaw = np.asarray(yaw)

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Stack along last dimension
    return np.stack((w, x, y, z), axis=-1)


def load_and_edit_mjcf_object(
    input_xml_path,
    output_xml_path,
    mesh_name="my_mesh",
    mesh_file="screwdriver_1.obj",
    new_scale=(0.1, 0.1, 0.1),
    body_name="mesh_body",
    new_pos=(0.0, 0.0, 0.0),
    new_quat=None,
):
    """
    Load an MJCF file, find or create the specified mesh in <asset>,
    and find or create a corresponding <body> in <worldbody>.
    Then, edit the mesh scale, body position, and orientation (quat),
    and save the updated file.

    Args:
        input_xml_path (str): Path to the original MJCF (XML).
        output_xml_path (str): Where to save the modified MJCF.
        mesh_name (str): Name attribute for the <mesh> and <geom>.
        mesh_file (str): File attribute for the <mesh> (obj, stl, ply, etc.).
        new_scale (tuple): (sx, sy, sz) scale values for the <mesh>.
        body_name (str): Name of the <body> that references the mesh.
        new_pos (tuple): (x, y, z) position for the <body>.
        new_quat (tuple or None): (qw, qx, qy, qz) orientation for the <body>.
                                  If None, orientation is unchanged (or not added).
    """
    # 1) Parse the XML
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    # 2) Find or create the <asset> element
    asset_elem = root.find("asset")
    if asset_elem is None:
        asset_elem = ET.SubElement(root, "asset")

    # 3) Find or create the specific <mesh> element by name
    mesh_elem = None
    for elem in asset_elem.findall("mesh"):
        if elem.get("name") == mesh_name:
            mesh_elem = elem
            break

    if mesh_elem is None:
        # If it doesn't exist, create it
        mesh_elem = ET.SubElement(
            asset_elem, "mesh", {"name": mesh_name, "file": mesh_file, "scale": " ".join(map(str, new_scale))}
        )
    else:
        # Update file path and scale if it already exists
        mesh_elem.set("file", mesh_file)
        mesh_elem.set("scale", " ".join(map(str, new_scale)))

    # 4) Find or create the <worldbody> element
    worldbody_elem = root.find("worldbody")
    if worldbody_elem is None:
        worldbody_elem = ET.SubElement(root, "worldbody")

    # 5) Find or create the <body> with the specified name
    body_elem = None
    for elem in worldbody_elem.findall("body"):
        if elem.get("name") == body_name:
            body_elem = elem
            break

    if body_elem is None:
        # If the body doesn't exist, create it
        body_elem = ET.SubElement(worldbody_elem, "body", {"name": body_name, "pos": " ".join(map(str, new_pos))})
    else:
        # Update its position
        body_elem.set("pos", " ".join(map(str, new_pos)))

    # Optionally set a new orientation (quat)
    if new_quat is not None:
        # You can choose 'quat' or 'euler' attribute. MuJoCo typically uses 'quat'.
        body_elem.set("quat", " ".join(map(str, new_quat)))

    # 6) Find or create the <geom> within this body that references the mesh
    mesh_geom_elem = None
    for elem in body_elem.findall("geom"):
        # Check if it's a mesh geom referencing our mesh name
        if elem.get("type") == "mesh" and elem.get("mesh") == mesh_name:
            mesh_geom_elem = elem
            break

    if mesh_geom_elem is None:
        # If there's no geom referencing our mesh, create one
        mesh_geom_elem = ET.SubElement(
            body_elem,
            "geom",
            {
                "name": f"{mesh_name}_geom",
                "type": "mesh",
                "mesh": mesh_name,
                # You can add more attributes as needed, e.g. friction, condim, etc.
            },
        )

    # 7) Save the modified XML back to disk
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


def edit_initial_ee_pose(filename, x, y, z, roll, pitch, yaw, body_name="palm"):
    """
    Open the given XML file, find the specified body, and update its pose.

    Args:
        filename (str): Path to the XML file to edit.
        x, y, z (float): Initial position in meters.
        roll, pitch, yaw (float): Orientation in degrees.
        body_name (str): Name of the body to edit (default is "palm").
    """
    # Load and parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()

    # Find the specified body element
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"Body with name '{body_name}' not found in the file.")

    # Update the position
    body.set("pos", f"{x} {y} {z}")

    # Convert Euler angles to quaternion
    w, qx, qy, qz = euler_to_quaternion_numpy(roll, pitch, yaw)
    body.set("quat", f"{w} {qx} {qy} {qz}")

    # Save the modified XML back to the file
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"Updated '{body_name}' pose in {filename}:")
    print(f"  Position: {x}, {y}, {z}")
    print(f"  Quaternion: {w}, {qx}, {qy}, {qz}")


# ------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    import os
    import sys

    parent_dir = "/home/lou/Desktop/mjcf/wonik_allegro/"
    input_mjcf = os.path.join(parent_dir, "scene_screwdriver.xml")

    output_mjcf = os.path.join(parent_dir, "scene_screwdriver.xml")

    # pose of object from semantic in real world (xyz, euler) in foundation pose
    scale = [0.15, 0.15, 0.15]
    # the default pose is the coordinate from blender to mujoco
    loaded_pose = (0, 0, 0.32 * scale[2], np.pi / 2, 0, 0)
    mesh_name = "my_mesh"
    mesh_file = "screw_driver_fix.obj"
    body_name = "mesh_body"

    # In 3D graphics and robotics, the RGB colors often correspond to the following axes:
    # - Red   -> X-axis (Roll)
    # - Green -> Y-axis (Pitch)
    # - Blue  -> Z-axis (Yaw)

    # Euler angles (roll, pitch, yaw) are rotations about these axes:
    # - Roll  is rotation about the X-axis (Red)
    # - Pitch is rotation about the Y-axis (Green)
    # - Yaw   is rotation about the Z-axis (Blue)

    # Example:
    # roll  = rotation about Red (X-axis)
    # pitch = rotation about Green (Y-axis)
    # yaw   = rotation about Blue (Z-axis)
    euler_angle = loaded_pose[3:]
    pos = loaded_pose[:3]

    quaternion = euler_to_quaternion_numpy(
        euler_angle[0], euler_angle[1], euler_angle[2]
    )  # euler, torch or pypose based on different implementation

    # we set the object as the free joint while the hand as the boundary condition

    load_and_edit_mjcf_object(
        input_xml_path=input_mjcf,
        output_xml_path=output_mjcf,
        mesh_name=mesh_name,
        mesh_file=mesh_file,
        new_scale=scale,  # example new scale
        body_name=body_name,
        new_pos=pos,  # example new position
        new_quat=quaternion,  # identity quaternion, or any orientation
    )

    input_filename = os.path.join(parent_dir, "right_hand.xml")
    initial_x, initial_y, initial_z = 0.1, 0, 0.1  # Position (in meters)
    initial_roll, initial_pitch, initial_yaw = np.pi / 2, np.pi, 0  # Orientation (in degrees)

    # Edit the end effector initial pose
    edit_initial_ee_pose(
        filename=input_filename,
        x=initial_x,
        y=initial_y,
        z=initial_z,
        roll=initial_roll,
        pitch=initial_pitch,
        yaw=initial_yaw,
    )

    print(f"Modified MJCF saved to: {output_mjcf}")
