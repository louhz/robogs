# robogs official release



We current support structure from motion toolset colmap and glomap, please follow the offical repo for installation and running instruction

this repo contains information necessary for asset creation and 4d rendering


# Installation 

Please install the latest torch version that is compatible with your device, we recommand torch 2.1 and cuda 11.8 later and earlier than cuda 12.8(we never test on the 50 series cuda device)

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

step 1: a monocular 360 video

Data Folder : ${datasetfolder} = sample_data
```shell
python robogs/vis/video2image.py -v sample_data/<video_path> -o sample_data/<image_output_directory> --num-frames <frame_count>
```
step 2: run struture from motion obtain features and camera pose

running colmap given the extracted images : https://colmap.github.io/install.html

step 3: run gsplat_trainer

Python Debugger: gsplat_trainer

step 4: run stable normal and save normal images to the normals folder

input the extracted images and pass it to stablenormal: https://huggingface.co/spaces/Stable-X/StableNormal

step 5: extract Guassian Splat, view and edit it in the supersplat viewer: https://superspl.at/editor/
Python Debugger: export ply

step 6: extract mesh

run the command in the launch.json
Python Debugger: trainmesh
Python Debugger: extractmesh

step 7: align the scene

please seen this demonstration video

step 8: Semantic label it or crop by manual 

please seen this demonstration video

step 9: assign id

edit the class id and run robogs/assign.py

step 10: fix the hyperparameter like mdh or phyiscal property

please seen : Python Debugger: debug and this demo video


step 11: produce urdf(mjcf)

clean up the bottom of object mesh

python robogs/mesh_util/fixbot.py -i input_mesh.stl -o output_mesh.stl

load it to the scene urdf(mjcf)



step 12: simulation and render


Python Debugger: 4drender