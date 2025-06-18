import mujoco  as mj

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import imageio
import cv2


def apply_action_and_step(mj_model: mj.MjModel, mj_data: mj.MjData, action,robot_action):
    """
    Places 'action' into mj_data.ctrl, then calls mj_step for one timestep.
    If you want continuous torque control or position servo, ensure your XML
    actuators are defined for these joints.
    """
    mj_data.ctrl[:7] = robot_action
    mj_data.ctrl[7:] = action
    mj.mj_step(mj_model, mj_data)




def main():

    mj_file='/home/haozhe/Dropbox/physics/franka_leap_demo/scene_nuetella_render.xml'
    # Load the MuJoCo model
    mj_model = mj.MjModel.from_xml_path(mj_file)
    # Create a MuJoCo data structure
    mj_data = mj.MjData(mj_model)
   
    # Set the initial state
    # i have a key frame in this mj_model as the home_pose

    home_pose_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, "home_pose")
    mj.mj_resetDataKeyframe(mj_model, mj_data, home_pose_id)

    robot_action = np.array([-0.435, 0.564, 0 ,-2.05, 1.19, 1.34, -1.85])
    
    robot_action_lift=np.array([-0.435, 0.2, 0 ,-2.05, 1.19, 1.34, -1.85])
    # arm pose 1
    # (-4,17,1,-119,1,137,-10)

    # arm pose 2
    # (-27,4,4,-92,86,68,-105)

    # arm pose 3
    # (-28,8,18,-112,78,82,-122)
    action_array=np.array([ 1.07 , 0.2,  0.4  , 0.809 , 0.8,  0.272,
    0.862,   0.21,  0.8, 0.408, 0.822 ,  0.857,
    2.1, 0.57, 0.07 , 0.254 ])


    # old_min, old_max = -1.47, 1.47
    # new_min, new_max = -2.5, 2.5

    # # # Vectorized scaling:
    # raw_action = new_min + (action_array - old_min) * (new_max - new_min) / (old_max - old_min)
    raw_action=action_array

    duration = 4  # seconds
    framerate = 60  # frames per second
    sim_steps = int(duration * framerate)
    # For rendering to frames (optional):
    frames = []
    height, width = 480, 640

    with mj.Renderer(mj_model, height=height, width=width) as renderer:
        for i in range(sim_steps):
            # Only apply the lift action after step 100
            if i < 100:
                # e.g., continue "grasp" action or do nothing
                apply_action_and_step(mj_model, mj_data, raw_action, robot_action)
            else:
                # after 100 steps, lift the arm
                apply_action_and_step(mj_model, mj_data, raw_action, robot_action_lift)

            # Render scene from camera "render"
            renderer.update_scene(mj_data, camera="render")
            pixels = renderer.render()
            frames.append(pixels)

    # Print final info
    print("Final sim time:", mj_data.time)
    print("Number of contacts at end:", mj_data.ncon)

    # Optional: Show video
    show_video(frames, fps=framerate)

    save_video(frames, fps=framerate, filename="my_simulation_video_nuetella.mp4")

def show_video(frames, fps):
    """Display frames using Matplotlib."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    plt.show()

def save_video(frames, fps, filename="output.mp4"):
    """
    Saves video from a list of frames using imageio.

    :param frames: List of frames (numpy arrays).
    :param fps: Frames per second.
    :param filename: Output file name with extension (e.g., "output.mp4").
    """
    with imageio.get_writer(filename, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

if __name__ == "__main__":
    main()
