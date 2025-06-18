from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch


# numpy version
def filter_with_semantic(semantic_id,assigned_ids,xyz,opacities,scales,features_extra,rots,features_dc,index=0):
    semantic_id_ind=(semantic_id==assigned_ids[index]).reshape(-1)
    semantic_id_ind_sam=semantic_id[semantic_id==assigned_ids[index]].reshape(-1,1)
        
    select_xyz= np.array(xyz[semantic_id_ind])
    select_opacities=  np.array(opacities[semantic_id_ind])
    select_scales =  np.array(scales[semantic_id_ind])
    select_features_extra =  np.array(features_extra[semantic_id_ind])
    select_rotation =  np.array(rots[semantic_id_ind])
    select_feature_dc =  np.array(features_dc[semantic_id_ind])


    return select_xyz,select_opacities,select_scales,select_features_extra,select_rotation,select_feature_dc,semantic_id_ind_sam





# torch version

def filter_with_semantic_torch(
    semantic_id: torch.Tensor,
    assigned_ids: torch.Tensor,
    xyz: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    features_extra: torch.Tensor,
    rots: torch.Tensor,
    features_dc: torch.Tensor,
    index: int = 0
):
    """
    Filters tensors based on matching semantic IDs.

    Args:
        semantic_id (torch.Tensor): 1D tensor of semantic IDs.
        assigned_ids (torch.Tensor): 1D tensor of possible assigned IDs.
        mark_id (torch.Tensor): (Unused in this example.)
        xyz (torch.Tensor): 2D tensor (N x D) of positions.
        opacities (torch.Tensor): 1D or 2D tensor of opacities.
        scales (torch.Tensor): 1D or 2D tensor of scales.
        features_extra (torch.Tensor): 2D tensor of additional features.
        rots (torch.Tensor): 2D or 3D tensor of rotations.
        features_dc (torch.Tensor): 2D tensor of some other features.
        index (int, optional): Index into assigned_ids; default is 0.

    Returns:
        tuple: (select_xyz, select_opacities, select_scales, select_features_extra,
                select_rotation, select_feature_dc, semantic_id_ind_sam)
    """

    # Boolean mask for elements where semantic_id == assigned_ids[index]
    semantic_id_ind = (semantic_id == assigned_ids[index]).view(-1)

    # Collect the actual matched semantic IDs in a column vector (N x 1)
    semantic_id_ind_sam = semantic_id[semantic_id_ind].view(-1, 1)

    # Filter each tensor by the boolean mask
    select_xyz = xyz[semantic_id_ind]
    select_opacities = opacities[semantic_id_ind]
    select_scales = scales[semantic_id_ind]
    select_features_extra = features_extra[semantic_id_ind]
    select_rotation = rots[semantic_id_ind]
    select_feature_dc = features_dc[semantic_id_ind]

    return (
        select_xyz,
        select_opacities,
        select_scales,
        select_features_extra,
        select_rotation,
        select_feature_dc,
        semantic_id_ind_sam
    )


