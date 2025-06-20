# Robo-gs official release
This repo is updated for the gsplat 1.0 and newest mesh reconstruction method: 2dgs+ normal constraint

Please follow the instruction and generate the digital asset, merge it with the policy training and sim2real deployment

# installation
```shell
pip install -r requirement.txt
```
this will install all the real2sim necessary information other than stablenormal and colmap

Please also make sure ffmpeg is installed in your system by: sudo apt install ffmpeg


# Reconstructed Asset:


Take a 360 surrended video, and run the Gaussian Splat and Mesh Extraction to obtain the Digital Asset

https://drive.google.com/file/d/1VVj6VdYO2MdxuC6HpMQBdeaHmR8LsskF/view?usp=drive_link

This the result of the reconstructed object, and hope this can provide you some hint of how to do the real2sim video capture

spray/spray/results/train/ours_7000/spray.ply is the final cleaned result


# todo

fix the config and the dataloader structure


fix the gripper 

fix object with mujoco backend



