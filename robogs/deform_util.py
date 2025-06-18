

import torch

import numpy as np
from plyfile import PlyData, PlyElement
import torch
import torch.nn.functional as F
import os
import shutil

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh_torch(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate spherical harmonics (0 <= deg <= 4) at unit directions (dirs)
    using hardcoded SH polynomials, in PyTorch.

    Args:
        deg: int SH deg. 0-4 supported
        sh:  torch.Tensor [..., C, (deg+1)**2]
        dirs:torch.Tensor [..., 3] (unit directions)
    Returns:
        torch.Tensor [..., C]
    """
    assert 0 <= deg <= 4, "deg must be 0, 1, 2, 3, or 4"
    coeff = (deg + 1)**2
    assert sh.shape[-1] >= coeff, f"Not enough SH coefficients for deg={deg}"

    # Start with L0 term
    result = C0 * sh[..., 0]

    if deg > 0:
        x = dirs[..., 0:1]
        y = dirs[..., 1:2]
        z = dirs[..., 2:3]
        result = (result
                  - C1 * y * sh[..., 1]
                  + C1 * z * sh[..., 2]
                  - C1 * x * sh[..., 3])

        if deg > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (result
                      + C2[0] * xy * sh[..., 4]
                      + C2[1] * yz * sh[..., 5]
                      + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                      + C2[3] * xz * sh[..., 7]
                      + C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                # Placeholder example. Make sure to fill C3 with correct values.
                # The calls below match your original shape, but you must 
                # ensure your actual constants are correct.
                result = (result
                          + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                          + C3[1] * xy * z * sh[..., 10]
                          + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                          + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                          + C3[5] * z * (xx - yy) * sh[..., 14]
                          + C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    # Placeholder example for C4 expansions.
                    # Fill with your own coefficients for deg = 4.
                    result = (result
                              + C4[0] * xy * (xx - yy) * sh[..., 16]
                              + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                              + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                              + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                              + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                              + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                              + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                              + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                              + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])

    return result


def sh_values_torch(sh: torch.Tensor, dirs: torch.Tensor):
    """
    Example function that splits the directional evaluations
    into "bands" the way your snippet does. 
    This is just to show a PyTorch version of your approach
    for building sh_values_1, sh_values_2, etc.

    Args:
        sh:   torch.Tensor (N, C, 15)  (for deg=3 example)
        dirs: torch.Tensor (1, 1, 15, 3) - directions
    Returns:
        sh_values_1, sh_values_2, sh_values_3
    """
    N, C, _ = sh.shape

    # Example: we create torch zeros in the same shape:
    sh_values_1 = torch.zeros((N, C, 3, 3), device=sh.device, dtype=sh.dtype)
    sh_values_2 = torch.zeros((N, C, 5, 5), device=sh.device, dtype=sh.dtype)
    sh_values_3 = torch.zeros((N, C, 7, 7), device=sh.device, dtype=sh.dtype)

    # For band 1
    x = dirs[..., 0:3, 0]  # shape (1,1,3)
    y = dirs[..., 0:3, 1]
    z = dirs[..., 0:3, 2]
    sh_values_1[..., 0] = -C1 * y
    sh_values_1[..., 1] =  C1 * z
    sh_values_1[..., 2] = -C1 * x

    # For band 2
    x = dirs[..., 3:8, 0]
    y = dirs[..., 3:8, 1]
    z = dirs[..., 3:8, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    sh_values_2[..., 0] = C2[0] * xy
    sh_values_2[..., 1] = C2[1] * yz
    sh_values_2[..., 2] = C2[2] * (2.0 * zz - xx - yy)
    sh_values_2[..., 3] = C2[3] * xz
    sh_values_2[..., 4] = C2[4] * (xx - yy)

    # For band 3
    x = dirs[..., 8:15, 0]
    y = dirs[..., 8:15, 1]
    z = dirs[..., 8:15, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    sh_values_3[..., 0] = C3[0] * y * (3 * xx - yy)
    sh_values_3[..., 1] = C3[1] * xy * z
    sh_values_3[..., 2] = C3[2] * y * (4 * zz - xx - yy)
    sh_values_3[..., 3] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    sh_values_3[..., 4] = C3[4] * x * (4 * zz - xx - yy)
    sh_values_3[..., 5] = C3[5] * z * (xx - yy)
    sh_values_3[..., 6] = C3[6] * x * (xx - 3 * yy)

    return sh_values_1, sh_values_2, sh_values_3


def sh_rotation_torch(sh: torch.Tensor, sh_dc: torch.Tensor, rotation: torch.Tensor):
    """
    Rotates spherical harmonics coefficients by sampling 
    directions, rotating them, and building a transform matrix.
    
    Args:
        sh:       (N, C, 15)  spherical harmonic coeffs (deg=3)
        sh_dc:    unused in snippet, presumably (N,C) with DC terms
        rotation: (3, 3)
    Returns:
        sh_rotation: rotated SHs (N, C, 15)
    """
    device = sh.device
    dtype = sh.dtype

    # Random directions on the unit sphere:
    dirs = torch.randn((15, 3), device=device, dtype=dtype)
    norm_dirs = torch.linalg.norm(dirs, dim=1, keepdim=True) + 1e-8
    dirs = dirs / norm_dirs  # (15,3)

    # Evaluate SH "bands" at original directions
    # Expand dims: (1,1,15,3)
    sh_values_1, sh_values_2, sh_values_3 = sh_values_torch(
        sh, dirs.view(1, 1, 15, 3)
    )

    # Rotate directions
    dirs_rot = dirs @ rotation.T  # (15,3)
    sh_values_1_rot, sh_values_2_rot, sh_values_3_rot = sh_values_torch(
        sh, dirs_rot.view(1, 1, 15, 3)
    )

    # We want to solve sh_values_1 @ X = sh_values_1_rot
    # So X = pinv(sh_values_1) @ sh_values_1_rot, shape must be broadcast carefully
    # pinv is done per (N,C) "slice" if needed, so confirm shapes carefully.
    # We'll assume dimension (N,C,3,3), so pinv must be done per row. 
    # A naive approach is to do it in a for-loop or flatten dims.

    # Here is a naive example:
    sh_transform_1 = torch.linalg.pinv(sh_values_1) @ sh_values_1_rot
    sh_transform_2 = torch.linalg.pinv(sh_values_2) @ sh_values_2_rot
    sh_transform_3 = torch.linalg.pinv(sh_values_3) @ sh_values_3_rot

    # Apply transform
    sh_rot = torch.zeros_like(sh)
    # shape (N,C,1,3) @ (N,C,3,3) -> (N,C,1,3), then squeeze
    sh_rot[..., 0:3] = (sh[..., None, 0:3] @ sh_transform_1)[..., 0, :]
    sh_rot[..., 3:8] = (sh[..., None, 3:8] @ sh_transform_2)[..., 0, :]
    sh_rot[..., 8:15] = (sh[..., None, 8:15] @ sh_transform_3)[..., 0, :]

    return sh_rot


def RGB2SH_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Example conversion from RGB to SH domain (DC = 0.5).
    """
    return (rgb - 0.5) / C0


def SH2RGB_torch(sh: torch.Tensor) -> torch.Tensor:
    """
    Example conversion back from SH domain to RGB.
    """
    return sh * C0 + 0.5


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
