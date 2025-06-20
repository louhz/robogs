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

to view the 4d render result
And change the camera view or the time frame, control signal to see the editing effect.

## Old version repo
This old version is build upon gsplat 0.7 and old version nerfstudio,
if you want to view the result of the demo shows in the paper, please look into this
https://github.com/RoboOmniSim/Robostudio/tree/main




# Create your own digital Asset

##  step 1: a monocular 360 video

example video: https://drive.google.com/drive/folders/1dCbJDBsMVjn15Ka24NKPwzptnwVoCqYI?usp=sharing

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
python train.py -s $data_path$ -r 2 --contribution_prune_ratio 0.5 --lambda_normal_prior 1 --lambda_dist 10 --densify_until_iter 3000 --iteration 7000 -m $mesh_result_path$ --w_normal_prior normals
python render.py -s $data_path$ -m $mesh_result_path$
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
Python Debugger: debug
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

## Step 12: Simulation and Rendering

* Load URDF/MJCF for simulation:

```
Python Debugger: 4drender
```

## Step 13: Physics-aware Rendering

* Perform FK, IK, collision detection.
* Execute 4D physics-aware rendering.
