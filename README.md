# robogs official release



We current support structure from motion toolset colmap and glomap, please follow the offical repo for installation and running instruction

this repo contains information necessary for asset creation and 4d rendering


# Installation 

Please install the latest torch version that is compatible with your device, we recommand torch 2.1 and cuda 11.8 later and earlier than cuda 12.8(we never test on the 50 series cuda device)

torch installation: pip3 install torch torchvision torchaudio(torch 2.7.1+cuda12.6)

Then cd in to robogs and follow the detail installation instruction


# running command

The key running command are stored in the launch.json, please run it with the vscode debugger

You just need to replace the parent folder of your data 

You can look into either the sample_data(franka arm+gripper+object)or the demo data(franka arm+allegro hand+object) 

## Example Data Format

run the Python Debugger: 4drender

please download this folder for the digital asset we create: https://www.dropbox.com/scl/fo/3rg66l348iyureo8amcen/AFm3SptGyT93fyFaXQghz-g?rlkey=ex134hgtuzzpog63z2d6t7mq5&st=ft3dqvcd&dl=0

This folder has the 4d rendering result:https://www.dropbox.com/scl/fo/pr3wh9431hqzrgi4conni/AGVedHBAc6riFiU46QZ8pQo?rlkey=mlohuopohrtkxppta80e2npv6&st=to0m9a0i&dl=0

You can view the 4d render result and change the camera view or the time frame, control signal to see the editing effect.



## Old version repo
This old version is build upon gsplat 0.7 and old version nerfstudio,
if you want to view the result of the demo shows in the paper, please look into this
https://github.com/RoboOmniSim/Robostudio/tree/main




# Create your own digital Asset

##  step 1: a monocular 360 video

sample_data: https://drive.google.com/drive/folders/1dCbJDBsMVjn15Ka24NKPwzptnwVoCqYI?usp=sharing

demo_data: https://drive.google.com/file/d/1hMgGnJQXrdtUnP0CDaNcqFa15qR-hLnm/view?usp=sharing

Data Folder : ${datasetfolder} = sample_data
```shell
python robogs/vis/video2image.py -v sample_data/<video_path> -o sample_data/<image_output_directory> --num-frames <frame_count>
```
##  step 2: run struture from motion obtain features and camera pose

running colmap given the extracted images : https://colmap.github.io/install.html

##  step 3: run gsplat_trainer

Python Debugger: gsplat_trainer

##  step 4: run stable normal and save normal images to the normals folder

input the extracted images and pass it to stablenormal: https://huggingface.co/spaces/Stable-X/StableNormal


# Gaussian Splat and Mesh Processing Pipeline

## Step 5: Extract Gaussian Splats, Mesh, and View

**Gaussian Splat Extraction**

* Data Folder: \`\$data\_path\$

```
Python Debugger: export ply
```

* View and edit Gaussian Splats: [SuperSplat Viewer](https://superspl.at/editor/)

**Mesh Extraction**


* Extract mesh and render:

```shell
python robogs/meshrecon/train.py -s $data_path$ -r 2 --contribution_prune_ratio 0.5 --lambda_normal_prior 1 --lambda_dist 10 --densify_until_iter 3000 --iteration 7000 -m $mesh_result_path$ --w_normal_prior normals
python robogs/meshrecon/render.py -s $data_path$ -m $mesh_result_path$
```

or run the launch.json

trainmesh
extractmesh


## Step 6: Align Scene

* Refer to demonstration video for alignment.

## Step 7: Segment and Semantic Label

* Watch segmentation and labeling demonstration.
* SAM-based segmentation (coming soon).

## Step 8: Assign ID

* Assign custom IDs:

```shell
python robogs/assign.py
```

## Step 9: Adjust Kinematics and Dynamics

* Fine-tune MDH and physical properties.

```
run the launch.json: debug
```

* Check associated demonstration video.

## Step 10: Coordinate and Scale Alignment

* Recenter and reorient Gaussian Splats and mesh.
* Keep alignment vectors.
* Perform automatic or ICP-based scale registration.

## Step 11: Generate URDF/MJCF

* Clean mesh bottom:

```shell
python robogs/mesh_util/fixbot.py -i input_mesh.stl -o output_mesh.stl
```

* Generate URDF/MJCF:

  * Compute bounding boxes and center of mass.
  * Infer physics parameters (VLM).

```shell
python robogs/mesh_util/generate_mjcf.py \
    -o <output_mjcf_path.xml> \
    -s <path_to_seed.ply> \
    -m <path_to_original_scene.xml> \
    --raw_image <path_to_raw_rgb_image.png> \
    --seg_image <path_to_segmentation_image.png>
```
The sample mjcf are stored in the franka_leap and franka_robotiq
## Step 12: Simulation and Rendering

* Load URDF/MJCF for simulation:
After you have the mjcf, please also be careful about the mjcf joint angle limit between sim and real


The scene should be align with the real world and able for rendering and simulation 

examples: mjcf_asset/franka_robotiq/scene_cup_gripper.xml

and see the keyframe for the similar result between sim and real. 

```
run the launch.json : 4drender
```

<!-- ## Step 13: Physics-aware Rendering

* Perform FK, IK, collision detection.
* Execute 4D physics-aware rendering. -->


if you find this work is helpful please cite this:
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