#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def edge_aware_curvature_loss(I, D, mask=None):
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device) / 4

    # Compute derivatives of D
    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])

    # Compute derivatives of I
    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x, padding=1) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y, padding=1) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    # Compute weights
    weights_x = (dI_dx - 1) ** 500
    weights_y = (dI_dy - 1) ** 500

    # Compute losses
    loss_x = torch.abs(dD_dx) * weights_x
    loss_y = torch.abs(dD_dy) * weights_y

    # Apply mask to losses
    if mask is not None:
        # Ensure mask is on the correct device and has correct dimensions
        mask = mask.to(I.device)
        loss_x = loss_x * mask
        loss_y = loss_y * mask

        # Count valid pixels
        valid_pixel_count = mask.sum()

        # Compute the mean loss only over valid pixels
        if valid_pixel_count.item() > 0:
            return (loss_x.sum() + loss_y.sum()) / valid_pixel_count
        else:
            # Handle the case where no valid pixels exist
            return torch.tensor(0.0, device=I.device, requires_grad=True)
    else:
        # If no mask is provided, calculate the mean over all pixels
        return (loss_x + loss_y).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def ms_l1_loss(network_output, gt, scales=[1, 2, 4]):
    total_loss = 0
    weights = [1.0, 0.5, 0.25]  # Weights for different scales, adjust as needed
    
    for scale, weight in zip(scales, weights):
        if scale == 1:
            # Original resolution
            total_loss += weight * l1_loss(network_output, gt)
        else:
            # Downsampled resolution
            scaled_output = F.interpolate(network_output, scale_factor=1/scale, mode='bilinear', align_corners=False)
            scaled_gt = F.interpolate(gt, scale_factor=1/scale, mode='bilinear', align_corners=False)
            total_loss += weight * l1_loss(scaled_output, scaled_gt)
    
    return total_loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l1_loss_appearance(image, gt_image, appearances, view_idx):
    if appearances is None:
        return l1_loss(image, gt_image)
    else:
        appearance_embedding = appearances.get_embedding(view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        crop_image = image[:, top:top+H, left:left+W]
        crop_gt_image = gt_image[:, top:top+H, left:left+W]
        
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
        
        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = appearances.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image
        return l1_loss(transformed_image, crop_gt_image)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # C1 = C2 = 0.01 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

