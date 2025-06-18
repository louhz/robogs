import os
import sys

current_file_dir = os.path.dirname(__file__)


workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree as KDTree
import typing
from collections import OrderedDict
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



from vis.gsplat_trainer import Runner, Config
import torch
import gsplat
import tyro
# Extend your original config to include checkpoint and export path.
from dataclasses import dataclass
from gsplat.strategy import DefaultStrategy, MCMCStrategy

@dataclass
class ExportConfig(Config):
    # Path to the checkpoint file to load.
    # ckpt: str ='results/bike/ckpts/ckpt_29999_rank0.pt'
    ckpt : str =''
    # Path where the exported .ply file will be saved.
    # export_ply_path: str = "exported.ply"
    export_ply_path : str =''
    # data_dir: str='test/home'
    data_dir : str = ''


def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # nerfstudio_version = "nerfstudio"
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            # ply_file.write(f"comment Generated by Nerstudio {nerfstudio_version}\n".encode())
            ply_file.write(b"comment Vertical Axis: z\n")
            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

def export_ply_main(cfg: ExportConfig):
    # Create a runner (using single-GPU / single-rank settings)
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(cfg.ckpt, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = checkpoint["splats"][k]

    # Extract parameters needed for export
    means = runner.splats["means"].detach().cpu().numpy()
    quats = runner.splats["quats"].detach().cpu().numpy()
    scales = runner.splats["scales"].detach().cpu().numpy()
    opacities = runner.splats["opacities"].detach().cpu().numpy()
    opacities = opacities.reshape(opacities.shape[0], 1)

    # Extract color information
    f_dc = runner.splats["sh0"].detach().cpu().numpy().reshape(means.shape[0], 3, 1)

    count = means.shape[0]
    f_rest = runner.splats["shN"].transpose(1, 2).contiguous().detach().cpu().numpy()
    f_rest = f_rest.reshape((count,-1))
    rotation = quats

    # Organize data into an ordered dictionary for PLY export
    
    map_to_tensors = OrderedDict()
    map_to_tensors["x"] = means[:, 0].reshape(count,1)
    map_to_tensors["y"] = means[:, 1].reshape(count,1)
    map_to_tensors["z"] = means[:, 2].reshape(count,1)
    
    for i in range(4):
        map_to_tensors[f"rot_{i}"] = rotation[:, i, None]
    for i in range(3):
        map_to_tensors[f"scale_{i}"] = scales[:, i, None]
    map_to_tensors["opacity"] = opacities

    # Store color information
    for i in range(f_dc.shape[1]):
        map_to_tensors[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[-1]):
        map_to_tensors[f"f_rest_{i}"] = f_rest[:, i, None]

    # Export to PLY
    write_ply(cfg.export_ply_path, count, map_to_tensors)
    print(f"PLY exported to {cfg.export_ply_path}")


def export_ply_sam(cfg: ExportConfig,semantic_info):
    # Create a runner (using single-GPU / single-rank settings)
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(cfg.ckpt, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = checkpoint["splats"][k]

    # Extract parameters needed for export
    means = runner.splats["means"].detach().cpu().numpy()
    quats = runner.splats["quats"].detach().cpu().numpy()
    scales = runner.splats["scales"].detach().cpu().numpy()
    opacities = runner.splats["opacities"].detach().cpu().numpy()
    opacities = opacities.reshape(opacities.shape[0], 1)
    semantic=semantic_info.reshape(opacities.shape[0],1)
    # Extract color information
    f_dc = runner.splats["sh0"].detach().cpu().numpy().reshape(means.shape[0], 3, 1)

    count = means.shape[0]
    f_rest = runner.splats["shN"].transpose(1, 2).contiguous().detach().cpu().numpy()
    f_rest = f_rest.reshape((count,-1))
    rotation = quats

    # Organize data into an ordered dictionary for PLY export
    
    map_to_tensors = OrderedDict()
    map_to_tensors["x"] = means[:, 0].reshape(count,1)
    map_to_tensors["y"] = means[:, 1].reshape(count,1)
    map_to_tensors["z"] = means[:, 2].reshape(count,1)
    
    for i in range(4):
        map_to_tensors[f"rot_{i}"] = rotation[:, i, None]
    for i in range(3):
        map_to_tensors[f"scale_{i}"] = scales[:, i, None]
    map_to_tensors["opacity"] = opacities

    # Store color information
    for i in range(f_dc.shape[1]):
        map_to_tensors[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(f_rest.shape[-1]):
        map_to_tensors[f"f_rest_{i}"] = f_rest[:, i, None]
    map_to_tensors['semantic_id']= semantic
    # Export to PLY
    write_ply(cfg.export_ply_path, count, map_to_tensors)
    print(f"PLY exported to {cfg.export_ply_path}")

if __name__ == '__main__':
    # Parse the configuration from command-line.
    configs = {
        "default": (
            "Gaussian splatting extraction for novel view and editing.",
            ExportConfig(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    export_ply_main(cfg)