import sys
import os
current_file_dir = os.path.dirname(__file__)


workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)

import warnings
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from fused_ssim import fused_ssim
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
    generate_360_path,
    generate_ellipse_path_y,
    generate_ellipse_path_x
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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vis.gsplat_trainer import Config, Parser, Dataset

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam



from robogs.engine.mdh.arm_deform import deform_arm,deform_finger,deform_object,deform_scene_combined,deform_arm_only,deform_finger_only



import gc
from assign import load_ply_sam, load_ply, save_ply_sam

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    


    def rasterize_splats_test(
        self,

        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,

        sh_degree: int = 3,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        # scales = self.splats["scales"]  # [N, 3]
        # opacities = self.splats["opacities"]  # [N,]

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=(False),
            sparse_grad=False,
            rasterize_mode='classic',
            distributed=False,
            camera_model='pinhole',
            sh_degree=sh_degree,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)



            
        # Helper function for rendering
    def render_step(self, xyz_out, rots_out, scales_out, opacities_out, fdc_out, fextra_out, device, camtoworlds, Ks, width, height, masks, cfg, stage, step, 
                    frame,
                    out_path):
        self.splats["means"] = torch.from_numpy(xyz_out).to(device).float()
        self.splats["quats"] = torch.from_numpy(rots_out).to(device)
        self.splats["scales"] = torch.from_numpy(scales_out).to(device)
        self.splats["opacities"] = torch.from_numpy(opacities_out).to(device).view(-1)

        self.splats['sh0'] = torch.from_numpy(fdc_out).to(device)
        self.splats['shN'] = torch.from_numpy(fextra_out).transpose(1, 2).to(device)

        renders, _, _ = self.rasterize_splats_test(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=cfg.sh_degree,
            masks=masks,
        )

        colors = torch.clamp(renders[..., :3], 0.0, 1.0)
        canvas = colors.squeeze(0).cpu().numpy()
        canvas = (canvas * 255).astype(np.uint8)

        imageio.imwrite(
            f"{out_path}/{frame}_{stage}_step{step:03d}.png",
            canvas
        )

        # Cleanup
        keys_to_delete = ['means', 'quats', 'scales', 'opacities', 'sh0', 'shN']
        for key in keys_to_delete:
            if key in self.splats:
                del self.splats[key]

        torch.cuda.empty_cache()
        gc.collect()


    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", ply_file: str = None, outpath: str = None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        if outpath is None:
            raise ValueError("outpath parameter cannot be None")

        if os.path.exists(outpath):
            warnings.warn(f"Output path '{outpath}' already exists. Files may be overwritten.")
        else:
            os.makedirs(outpath)

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1
        )

        camtoworlds1 = torch.tensor(
            [[[-0.6800, 0.4781, -0.5559, 1.8009],
            [0.1233, -0.6728, -0.7295, 1.4379],
            [-0.7228, -0.5646, 0.3985, -0.9330],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'
        )

        camtoworlds2 = torch.tensor(
            [[[0.3487, 0.7220, -0.5976, 1.8779],
            [0.5092, -0.6813, -0.5259, 1.4589],
            [-0.7869, -0.1209, -0.6052, 1.3671],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'
        )

        camtoworlds3 = torch.tensor(            [[[-0.0473,  0.6528, -0.7560,  1.9752],
            [ 0.1960, -0.7361, -0.6479,  1.1878],
            [-0.9795, -0.1788, -0.0932,  0.0921],
            [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')

        data = trainloader.dataset[0]
        camtoworlds = camtoworlds3

        Ks = data["K"].to(device).view(1, 3, 3)
        pixels = data["image"].to(device) / 255.0
        pixels = pixels.view(1, pixels.shape[0], pixels.shape[1], 3)
        masks = data.get("mask", None)
        if masks is not None:
            masks = masks.to(device)

        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()

        xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = load_ply_sam(ply_file)

        stages = [
            # ("deform_arm", deform_arm),
            ("deform_hand_object", deform_scene_combined),
        ]
        idx=0
        for stage, deform_func in stages:
            for step, (xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out) in enumerate(
                deform_func(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id)
            ):
                self.render_step(
                    xyz_out, rots_out, scales_out, opacities_out, fdc_out,
                    fextra_out, device, camtoworlds, Ks, width, height, masks, cfg, stage, step,
                    frame=idx,
                    out_path=outpath
                )

    @torch.no_grad()
    def eval_traj(self, step: int, stage: str = "val", ply_file: str = None, outpath: str = None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        if outpath is None:
            raise ValueError("outpath parameter cannot be None")

        if os.path.exists(outpath):
            warnings.warn(f"Output path '{outpath}' already exists. Files may be overwritten.")
        else:
            os.makedirs(outpath)

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1
        )

        camtoworlds1 = torch.tensor(
            [[[-0.6800, 0.4781, -0.5559, 1.8009],
            [0.1233, -0.6728, -0.7295, 1.4379],
            [-0.7228, -0.5646, 0.3985, -0.9330],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device
        )

        camtoworlds2 = torch.tensor(
            [[[0.3487, 0.7220, -0.5976, 1.8779],
            [0.5092, -0.6813, -0.5259, 1.4589],
            [-0.7869, -0.1209, -0.6052, 1.3671],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device=device
        )


        # cfg.render_traj_path = "ellipse"
        # # Select trajectory generation mode
        # if cfg.render_traj_path == "interp":
        #     camtoworlds_all = generate_interpolated_path(
        #         np.concatenate([camtoworlds1.cpu().numpy(), camtoworlds2.cpu().numpy()], axis=0), n_interp=30
        #     )  # [N, 3, 4]

        # elif cfg.render_traj_path == "ellipse":
        #     height = (camtoworlds1[0, 2, 3].item() + camtoworlds2[0, 2, 3].item()) / 2
        #     camtoworlds_all = generate_ellipse_path_x(
        #         np.concatenate([camtoworlds1.cpu().numpy(), camtoworlds2.cpu().numpy()], axis=0), height=height
        #     )  # [N, 3, 4]

        # elif cfg.render_traj_path == "spiral":
        #     camtoworlds_all = generate_spiral_path(
        #         np.concatenate([camtoworlds1.cpu().numpy(), camtoworlds2.cpu().numpy()], axis=0),
        #         bounds=(self.parser.bounds * self.scene_scale),
        #         spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
        #     )
        # elif cfg.render_traj_path == "360":

        #     camtoworlds_all = generate_360_path(camtoworlds1.cpu().numpy(), camtoworlds2.cpu().numpy(), n_poses=30)

        # else:
        #     raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

        # # Append [0, 0, 0, 1] to make it 4x4 homogeneous matrices
        # num_frames = camtoworlds_all.shape[0]
        # extra_row = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, 1, 1))
        # camtoworlds_all = np.concatenate([camtoworlds_all, extra_row], axis=1)

        # camtoworlds_all = np.array([

        #     [[-0.8583,     0.3175,    -0.4032,     1.2881],
        #     [ 0.1889,    -0.5349,    -0.8236,     0.6618],
        #     [-0.4771,    -0.7830,     0.3989,    -0.9646],
        #     [ 0.0000,     0.0000,     0.0000,     1.0000]],

        #     [[-0.7511,  0.6106, -0.2508,  1.0183],
        #     [-0.4458, -0.7493, -0.4898,  0.9685],
        #     [-0.4870, -0.2560,  0.8350, -1.1460],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[-0.8619, -0.0139, -0.5070,  1.3026],
        #     [ 0.1413, -0.9665, -0.2140,  0.4874],
        #     [-0.4870, -0.2560,  0.8350, -0.9860],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[-0.8619, -0.0139, -0.5070,  1.5026],
        #     [ 0.1413, -0.9665, -0.2140,  0.6874],
        #     [-0.4870, -0.2560,  0.8350, -1.3860],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],


        #     [[-0.3442,  0.3883, -0.8548,  1.5266],
        #     [ 0.3306, -0.8020, -0.4974,  1.4852],
        #     [-0.8788, -0.4538,  0.1476,  0.1511],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[-0.0473,  0.6528, -0.7560,  1.9352],
        #     [ 0.1960, -0.7361, -0.6479,  1.4278],
        #     [-0.9795, -0.1788, -0.0932,  0.4921],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[-0.2436,  0.6874, -0.6842,  1.8653],
        #     [ 0.2256, -0.6459, -0.7293,  1.2213],
        #     [-0.9433, -0.3320,  0.0022,  0.4927],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[ 0.4597,  0.5971, -0.6574,  1.7592],
        #     [ 0.5569, -0.7704, -0.3103,  1.3194],
        #     [-0.6918, -0.2235, -0.6866,  0.7731],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[ 0.2053,  0.8768, -0.4348,  1.7892],
        #     [ 0.5941, -0.4647, -0.6566,  1.6843],
        #     [-0.7777, -0.1236, -0.6163,  0.9211],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]],

        #     [[ 0.3530,  0.7045, -0.6157,  2.0995],
        #     [ 0.6081, -0.6729, -0.4213,  1.9781],
        #     [-0.7111, -0.2257, -0.6659,  1.5261],
        #     [ 0.0000,  0.0000,  0.0000,  1.0000]]
        # ])


        camtoworlds_all = np.array([
            [[-0.7046,  0.5506, -0.4476,  1.7585],
            [ 0.1688, -0.4826, -0.8594,  1.6100],
            [-0.6892, -0.6811,  0.2471, -1.1348],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.4920,  0.6141, -0.6171,  1.7157],
            [ 0.1032, -0.6627, -0.7418,  1.2324],
            [-0.8645, -0.4286,  0.2626, -0.5961],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.3731,  0.6325, -0.6788,  1.2763],
            [ 0.3412, -0.5868, -0.7343,  1.4565],
            [-0.8628, -0.5056,  0.0031, -0.2607],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.3832,  0.7394, -0.5536,  1.5750],
            [ 0.1648, -0.5350, -0.8286,  1.7061],
            [-0.9089, -0.4087,  0.0831,  0.2483],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.3442,  0.3883, -0.8548,  1.5266],
            [ 0.3306, -0.8020, -0.4974,  1.4852],
            [-0.8788, -0.4538,  0.1476,  0.1511],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.0473,  0.6528, -0.7560,  1.9752],
            [ 0.1960, -0.7361, -0.6479,  1.1878],
            [-0.9795, -0.1788, -0.0932,  0.0921],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[-0.2436,  0.6874, -0.6842,  1.8053],
            [ 0.2256, -0.6459, -0.7293,  1.2213],
            [-0.9433, -0.3320,  0.0022,  0.4927],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[ 0.4597,  0.5971, -0.6574,  1.7592],
            [ 0.5569, -0.7704, -0.3103,  1.3194],
            [-0.6918, -0.2235, -0.6866,  0.7731],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[ 0.2053,  0.8768, -0.4348,  1.7892],
            [ 0.5941, -0.4647, -0.6566,  1.6843],
            [-0.7777, -0.1236, -0.6163,  0.9211],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],

            [[ 0.3530,  0.7045, -0.6157,  2.0995],
            [ 0.6081, -0.6729, -0.4213,  1.9781],
            [-0.7111, -0.2257, -0.6659,  1.5261],
            [ 0.0000,  0.0000,  0.0000,  1.0000]]
        ])

    #     camtoworlds_all = np.array([[[-0.25571028,  0.77972345, -0.57152743,  1.31836274],
    #     [ 0.39014636, -0.45767218, -0.79895056,  1.80452242],
    #     [-0.88453269, -0.42727921, -0.18717478,  0.16107921],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.43921331,  0.67938284, -0.58781853,  1.43302102],
    #     [ 0.14771713, -0.5907887 , -0.79318873,  1.7451265 ],
    #     [-0.88615536, -0.43520992,  0.15912578, -0.04823827],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.49732169,  0.57743714, -0.64748551,  1.56980522],
    #     [ 0.24878255, -0.62006064, -0.74406454,  1.77739776],
    #     [-0.83113078, -0.53112253,  0.16471332,  0.20269549],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.51576104,  0.77329919, -0.3687803 ,  1.17929808],
    #     [-0.12149379, -0.49211716, -0.86200926,  1.56573627],
    #     [-0.84807417, -0.39978628,  0.34776591,  0.23541179],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.39897443,  0.6769695 , -0.61849147,  1.41568578],
    #     [ 0.26745189, -0.55925966, -0.78466434,  1.81371357],
    #     [-0.87709115, -0.47847772,  0.04207353,  0.32883805],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.38233272,  0.80622504, -0.45146746,  1.83579987],
    #     [ 0.14866627, -0.42855152, -0.89120252,  1.69892993],
    #     [-0.91198686, -0.40785387,  0.0439908 ,  0.18623443],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.262307  ,  0.8249378 , -0.50067221,  1.52788571],
    #     [ 0.0254255 , -0.51275126, -0.85816064,  1.67087734],
    #     [-0.96464946, -0.23783139,  0.11352379,  0.41782408],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.53967509,  0.74103498, -0.39952216,  1.50170509],
    #     [ 0.17496906, -0.36547319, -0.91422928,  1.72596123],
    #     [-0.82349051, -0.56329079,  0.06757858, -0.04091622],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.29614988,  0.74751285, -0.59457531,  1.9438858 ],
    #     [ 0.06392605, -0.60559177, -0.79320368,  1.58412952],
    #     [-0.95299985, -0.27291602,  0.13156038,  0.11324861],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.19208169,  0.82542507, -0.53082774,  1.44000863],
    #     [-0.07177363, -0.55126692, -0.83123602,  1.54734538],
    #     [-0.97875082, -0.12156579,  0.16513203,  0.19125809],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.26305695,  0.83521732, -0.4829214 ,  1.71218698],
    #     [ 0.02323765, -0.49492033, -0.86862759,  1.71445462],
    #     [-0.96450042, -0.23972048,  0.11078374,  0.03446797],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.41836549,  0.62174253, -0.66212276,  1.63024203],
    #     [ 0.21615332, -0.63988853, -0.7374418 ,  1.79517034],
    #     [-0.88218369, -0.45164023,  0.13331556,  0.48276494],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.25165108,  0.88500301, -0.39171599,  1.64485462],
    #     [ 0.14614766, -0.36535059, -0.91932573,  1.80019122],
    #     [-0.95671971, -0.28859769, -0.03740025,  0.57048758],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.47197387,  0.62002987, -0.62674048,  1.66820608],
    #     [ 0.25225014, -0.58620714, -0.76989029,  1.93897406],
    #     [-0.84475472, -0.52146347,  0.12027183,  0.26158314],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.26323427,  0.82613149, -0.49821128,  1.55663574],
    #     [ 0.26892283, -0.4331192 , -0.86028383,  1.72765158],
    #     [-0.92649243, -0.36043658, -0.1081538 ,  0.37506483],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.3595042 ,  0.74950878, -0.55587167,  1.15730131],
    #     [ 0.11370529, -0.55607344, -0.82331855,  1.92400556],
    #     [-0.92618996, -0.35919203,  0.11468762,  0.44121812],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.28145965,  0.8676621 , -0.40980843,  1.69256686],
    #     [ 0.25246314, -0.34506841, -0.90398571,  1.76111806],
    #     [-0.92576608, -0.35789702, -0.12192986,  0.4944598 ],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.28661045,  0.76399082, -0.57807653,  1.4496942 ],
    #     [ 0.2883756 , -0.50661075, -0.81251773,  1.93580344],
    #     [-0.91361587, -0.39957924, -0.07511638,  0.36642564],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.23171623,  0.78637993, -0.57263792,  1.60185292],
    #     [-0.10039689, -0.60484779, -0.7899871 ,  1.46058811],
    #     [-0.96758878, -0.12556176,  0.21910315,  0.17082856],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]],

    #    [[-0.25238779,  0.89429063, -0.36951952,  1.48528549],
    #     [ 0.13741901, -0.34488513, -0.92853124,  1.81465876],
    #     [-0.95781857, -0.28512895, -0.03584774,  0.06122255],
    #     [ 0.        ,  0.        ,  0.        ,  1.        ]]])

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)


        data = trainloader.dataset[0]

        Ks = data["K"].to(device).view(1, 3, 3)
        pixels = data["image"].to(device) / 255.0
        pixels = pixels.view(1, pixels.shape[0], pixels.shape[1], 3)
        masks = data.get("mask", None)
        if masks is not None:
            masks = masks.to(device)

        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()

        xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = load_ply_sam(ply_file)

        features_dc_use=torch.from_numpy(features_dc).transpose(1, 2).cpu().numpy()
        # Render for each camera in the generated trajectory
        for idx, c2w in enumerate(camtoworlds_all):
            c2w = c2w.unsqueeze(0)  # add batch dimension

            print(f"Rendering frame {idx+1}/{len(camtoworlds_all)}...")
            self.render_step(
                xyz, rots, scales, opacities, features_dc_use,
                features_extra, device, c2w, Ks, width, height, masks, cfg, stage, step,
                frame=idx,
                out_path=outpath
            )


    @torch.no_grad()
    def render_foreground(self, step: int, stage: str = "val", ply_file: str = None, outpath: str = None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        if outpath is None:
            raise ValueError("outpath parameter cannot be None")

        if os.path.exists(outpath):
            warnings.warn(f"Output path '{outpath}' already exists. Files may be overwritten.")
        else:
            os.makedirs(outpath)

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1
        )

        camtoworlds1 = torch.tensor(
            [[[-0.6800, 0.4781, -0.5559, 1.8009],
            [0.1233, -0.6728, -0.7295, 1.4379],
            [-0.7228, -0.5646, 0.3985, -0.9330],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'
        )

        camtoworlds2 = torch.tensor(
            [[[0.3487, 0.7220, -0.5976, 1.8779],
            [0.5092, -0.6813, -0.5259, 1.4589],
            [-0.7869, -0.1209, -0.6052, 1.3671],
            [0.0000, 0.0000, 0.0000, 1.0000]]], device='cuda:0'
        )

        camtoworlds3 = torch.tensor([
            [[-0.4920,  0.6141, -0.6171,  2.457],
            [ 0.1032, -0.6627, -0.7418,  2.324],
            [-0.8645, -0.4286,  0.2626, -0.5961],
            [ 0.0000,  0.0000,  0.0000,  1.0000]],], device='cuda:0')

        data = trainloader.dataset[0]
        camtoworlds = camtoworlds3

        Ks = data["K"].to(device).view(1, 3, 3)
        pixels = data["image"].to(device) / 255.0
        pixels = pixels.view(1, pixels.shape[0], pixels.shape[1], 3)
        masks = data.get("mask", None)
        if masks is not None:
            masks = masks.to(device)

        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()

        xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = load_ply_sam(ply_file)

        stages = [
            # ("deform_arm", deform_arm),
            ("deform_hand_object", deform_scene_combined),
        ]
        idx=0
        for stage, deform_func in stages:
            for step, (xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out) in enumerate(
                deform_func(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id)
            ):
                self.render_step(
                    xyz_out, rots_out, scales_out, opacities_out, fdc_out,
                    fextra_out, device, camtoworlds, Ks, width, height, masks, cfg, stage, step,
                    frame=idx,
                    out_path=outpath
                )
    @torch.no_grad()
    def exportdeform(self, step: int, stage: str = "val",ply_file: str = None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)




        camtoworlds1= torch.tensor([[[-0.6800,  0.4781, -0.5559,  1.8009],
                [ 0.1233, -0.6728, -0.7295,  1.4379],
                [-0.7228, -0.5646,  0.3985, -0.9330],
                [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')
                
        camtoworlds2 =torch.tensor([[[ 0.3487,  0.7220, -0.5976,  1.8779],
         [ 0.5092, -0.6813, -0.5259,  1.4589],
         [-0.7869, -0.1209, -0.6052,  1.3671],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device='cuda:0')


        data=trainloader.dataset[716]
        camtoworlds = data["camtoworld"].view(1, 4, 4).to(device)  # [1, 4, 4]
                        # add a translation to the camtoworlds

                        # frame 0 for the object side view
        # camtoworlds[:,:3, 3] += torch.tensor([1.5, 0.38, 0.4], device=device).view(1,3) # x,z,y in supersplat

        camtoworlds=camtoworlds2

            
        Ks = data["K"].to(device).view(1, 3, 3)  # [1, 3, 3]
        pixels = data["image"].to(device) / 255.0
        pixels=pixels.view(1, pixels.shape[0], pixels.shape[1],3)  # [1, H, W, 3]
        masks = data["mask"].to(device) if "mask" in data else None
        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()
        tic = time.time()


        # deform here
        xyz, features_dc, features_extra, opacities, scales, rots,semantic_id = load_ply_sam(ply_file)

        #


        # Save the deformed splats for export

        for step, (xyz_out, opacities_out, scales_out, fextra_out, rots_out, fdc_out, sem_out) in enumerate(
                deform_finger_only(xyz, features_dc, features_extra, opacities, scales, rots, semantic_id)
            ):
            # Sequence of scene deformation
            save_ply_sam(
                xyz=xyz_out,
                opacities= opacities_out,
                scale= scales_out,
                f_rest= fextra_out,
                rotation= rots_out,  
                f_dc= fdc_out,
                semantic_id= sem_out,
                path= f"/home/haozhe/Dropbox/rendering/asset/debug/deform_finger_onlystep{step:03d}.ply",
            )


    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")


    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()