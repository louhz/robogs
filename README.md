# Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm with Hybrid Representation

[![arXiv](https://img.shields.io/badge/ArXiv-2408.14873-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2408.14873) [![web](https://img.shields.io/badge/Web-Robostudio-blue.svg?style=plastic)](https://www.robostudioapp.com/) [![license](https://img.shields.io/badge/LICENSE-CC_BY--NC_4.0-white.svg?style=plastic)](https://github.com/louhz/robogs/blob/main/LICENSE)


> **Official Release**  
> We currently support *structure from motion* toolsets: **COLMAP** and **GLOMAP**.  
> Follow the documentation for asset creation and 4D rendering.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Example Data](#example-data)
- [Citation](#citation)

## 🚀 Installation

**[PyTorch](https://pytorch.org/get-started/previous-versions/)** (Recommended: PyTorch 2.1 + CUDA 11.8-12.6):
```bash
pip3 install torch torchvision torchaudio  # torch 2.7.1 with cuda 12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # torch 2.7.1 with cuda 11.8
```

> **⚠️ Note:** CUDA 11.8 or later is recommended, but not CUDA 12.8 or newer. 50-series CUDA devices are untested.

**Structure from Motion Tools**

- **[COLMAP Installation Guide](https://colmap.github.io/install.html)**
- **[GLOMAP Repository](https://github.com/GLOMAP/GLOMAP)** (follow their official instructions)

### Project Setup

Follow the detailed installation instructions in [robogs/installation.md](robogs/installation.md).

## 🎯 Quick Start

We provide several pre-configured debug configurations in `.vscode/launch.json` for different pipeline stages. To use these configurations, replace the data paths with your own data location in the launch.json file.

## 📦 Example Data

- **[Download Sample Data](https://drive.google.com/drive/folders/1dCbJDBsMVjn15Ka24NKPwzptnwVoCqYI?usp=sharing)**: Franka arm with parallel gripper and manipulated object
- **[Download Demo Data](https://drive.google.com/file/d/1hMgGnJQXrdtUnP0CDaNcqFa15qR-hLnm/view?usp=sharing)**: Franka arm with Robotiq hand and manipulated object
- **[Download Digital Assets](https://www.dropbox.com/scl/fo/pr3wh9431hqzrgi4conni/AGVedHBAc6riFiU46QZ8pQo?rlkey=mlohuopohrtkxppta80e2npv6&st=to0m9a0i&dl=0)**
- **[Download 4D Rendering Results](https://www.dropbox.com/scl/fo/3rg66l348iyureo8amcen/AFm3SptGyT93fyFaXQghz-g?rlkey=ex134hgtuzzpog63z2d6t7mq5&st=ft3dqvcd&dl=0)**

> **Tip:** You can interactively view 4D render results, change camera views, time frames, and control signals to see editing effects.

### Legacy Version
For results from the original paper (using GSplat 0.7 and old NeRFStudio), see the [Old Version Repository](https://github.com/RoboOmniSim/Robostudio).

## 🔄 Complete Pipeline

### Step 1: Capture Monocular 360° Video

Extract frames from video:
```bash
python robogs/vis/video2image.py \
    -v sample_data/<video_path> \
    -o sample_data/<image_output_directory> \
    --num-frames <frame_count>
```

### Step 2: Structure from Motion

Run COLMAP on the extracted images to obtain features and camera poses. See the [COLMAP documentation](https://colmap.github.io/install.html) for details.

### Step 3: GSplat Training

Train the Gaussian Splatting model:
```bash
python robogs/vis/gsplat_trainer.py \
    default \
    --data_dir sample_data \
    --data_factor 1 \
    --result-dir sample_data/gs_result_sfm
```

or if you want to launch with debugger, please refer to the configuration `Python Debugger: gsplat_trainer` in `.vscode/launch.json`.

### Step 4: Normal Map Generation

Use the [StableNormal Tool](https://github.com/Stable-X/StableNormal) to generate normal maps from the extracted images. Save the normal images to the `normals` folder.

### Step 5: Gaussian Splat & Mesh Processing

#### Gaussian Splat Extraction

Extract Gaussian Splat point clouds:
```bash
python robogs/vis/extract_ply.py \
    --input <trained_model_path> \
    --output <output_ply_path>
```

or if you want to launch with debugger, please refer to the configuration `Python Debugger: export ply` in `.vscode/launch.json`.

To view and edit Gaussian Splat, use the [SuperSplat Viewer](https://superspl.at/editor/).

#### Mesh Extraction

Train and render mesh:
```bash
python robogs/meshrecon/train.py \
    -s <data_path> \
    -r 2 \
    --contribution_prune_ratio 0.5 \
    --lambda_normal_prior 1 \
    --lambda_dist 10 \
    --densify_until_iter 3000 \
    --iteration 7000 \
    -m <mesh_result_path> \
    --w_normal_prior normals

python robogs/meshrecon/render.py \
    -s <data_path> \
    -m <mesh_result_path>
```

You can also use the VSCode debugger configurations `trainmesh` and `extractmesh` for these steps.

### Step 6: Scene Alignment

Align the reconstructed scene. Refer to the demonstration video for details.

### Step 7: Segmentation & Labeling

Segment and label the scene. See the demonstration video for details. (A SAM-based segmentation tool is coming soon.)

### Step 8: ID Assignment

Assign custom IDs:
```bash
python robogs/assign.py
```

### Step 9: Kinematics & Dynamics

Fine-tune MDH and physical properties using the `Python Debugger: debug` configuration in VSCode. See the demo video for more information.

### Step 10: Coordinate & Scale Alignment

- Recenter and reorient the Gaussian Splats and mesh.
- Keep alignment vectors.
- Perform automatic ICP-based scale registration.

### Step 11: URDF/MJCF Generation

Clean the mesh bottom:
```bash
python robogs/mesh_util/fixbot.py \
    -i input_mesh.stl \
    -o output_mesh.stl
```

Generate URDF/MJCF:
```bash
python robogs/mesh_util/generate_mjcf.py \
    -o <output_mjcf_path.xml> \
    -s <path_to_seed.ply> \
    -m <path_to_original_scene.xml> \
    --raw_image <path_to_raw_rgb_image.png> \
    --seg_image <path_to_segmentation_image.png>
```

Sample MJCF files are stored in the `franka_leap` and `franka_robotiq` folders.

### Step 12: Simulation & Rendering

> **⚠️ Important**: After generating MJCF, carefully check joint angle limits between simulation and real-world.

The scene should be aligned with the real world and ready for rendering and simulation. See `mjcf_asset/franka_robotiq/scene_cup_gripper.xml` for an example, and refer to the `Python Debugger: 4drender` configuration in VSCode for visualization.

## 📚 Citation

If you find this work helpful, please cite:

```bibtex
@misc{lou2024robogsphysicsconsistentspatialtemporal,
  title={Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm with Hybrid Representation}, 
  author={Haozhe Lou and Yurong Liu and Yike Pan and Yiran Geng and Jianteng Chen and Wenlong Ma and Chenglong Li and Lin Wang and Hengzhen Feng and Lu Shi and Liyi Luo and Yongliang Shi},
  year={2024},
  eprint={2408.14873},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2408.14873}, 
}
```