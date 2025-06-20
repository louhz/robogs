# use chatgpt to load the xml file, edit the mesh path, physics parameter and the relative location from the real2sim part



# the general step




# load the object and urdf position from foundation pose


# set it up use the openai api



# the major thing is setup the collision ball for mjcf and curobo


import os
import xml.etree.ElementTree as ET
import openai  # Only needed if you plan to use OpenAI API calls
import json

###############################################################################
# STEP 1: ChatGPT/OpenAI Prompting (Optional)
###############################################################################
def ask_chatgpt_for_edits(prompt_text, openai_api_key):
    """
    An example function that queries the OpenAI API to get suggestions
    about how to edit parameters. This is entirely optional.
    """
    openai.api_key = openai_api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0,
    )

    # Extract the assistant's reply
    reply = response["choices"][0]["message"]["content"]
    return reply

###############################################################################
# STEP 2: Load External Data (e.g., Foundation Pose, URDF positions)
###############################################################################
def load_foundation_pose(pose_path):
    """
    Load a foundation pose (e.g., object positions, URDF positions) from JSON or other format.
    For demonstration, we assume a JSON with structure:
    {
      "objects": [
        {
          "name": "box_obj",
          "position": [0.1, 0.2, 0.3],
          "rotation_degs": [0, 90, 0]
        },
        ...
      ]
    }
    """
    with open(pose_path, "r") as f:
        data = json.load(f)
    return data

###############################################################################
# STEP 3: Edit MJCF (MuJoCo XML) to Update Mesh Paths, Physics Params, etc.
###############################################################################
def edit_mjcf_xml(xml_path, output_path, foundation_data, real2sim_transform=None):
    """
    Load the MJCF file, update mesh paths, physics parameters, and add collision geometry.
    Save the updated file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1) Edit mesh paths
    for mesh in root.findall(".//mesh"):
        old_path = mesh.get("file")
        if old_path is not None:
            # Example: prepend a new directory or rename extension
            new_path = os.path.join("new_mesh_dir", os.path.basename(old_path))
            mesh.set("file", new_path)
            print(f"Updated mesh path from {old_path} to {new_path}")

    # 2) Edit physics parameters (e.g., friction, density, damping, etc.)
    for geom in root.findall(".//geom"):
        # Example: set friction to a certain value
        geom.set("friction", "0.8 0.1 0.1")  # Demo
        # You can also read from the foundation_data or real2sim_transform if needed

    # 3) Use real2sim_transform on object positions or body positions
    #    Suppose real2sim_transform is a function or matrix that converts real -> sim coords.
    for body in root.findall(".//body"):
        if body.get("name") in [obj["name"] for obj in foundation_data.get("objects", [])]:
            # Get the foundation pose for this object
            obj_data = next(o for o in foundation_data["objects"] if o["name"] == body.get("name"))
            # Apply real2sim transform if needed
            real_pos = obj_data["position"]  # e.g., [x, y, z]
            sim_pos = real2sim_transform(real_pos) if real2sim_transform else real_pos
            # Update the body pos in the MJCF
            old_pos = body.get("pos")
            new_pos = f"{sim_pos[0]} {sim_pos[1]} {sim_pos[2]}"
            body.set("pos", new_pos)
            print(f"Updated body '{body.get('name')}' position from {old_pos} to {new_pos}")

    # 4) Add a collision ball geometry (example)
    #    Let's assume we want a new body in the world with a sphere geom
    collision_body = ET.Element("body", {"name": "collision_ball", "pos": "0 0 0"})
    collision_geom = ET.SubElement(collision_body, "geom", {
        "type": "sphere",
        "size": "0.05",  # radius
        "density": "1000",
        "rgba": "1 0 0 1",  # red
        "contype": "1",
        "conaffinity": "1"
    })
    # Insert the new body at the top-level (or under a specific parent <body>)
    root.append(collision_body)
    print("Added collision_ball body with a sphere geom.")

    # Save updated XML
    tree.write(output_path)
    print(f"\nSaved updated MJCF to {output_path}")


###############################################################################
# STEP 4: MAIN
###############################################################################
def main():
    # Optional: put your OpenAI API key here
    openai_api_key = "YOUR_OPENAI_API_KEY"

    # Example usage of ChatGPT to get parameter suggestions (HIGH-LEVEL):
    # comment out if you do not want to use ChatGPT
    prompt_text = "What friction and density parameters are recommended for a plastic cube in a MuJoCo simulation?"
    suggestions = ask_chatgpt_for_edits(prompt_text, openai_api_key)
    print("\nChatGPT suggestions:\n", suggestions)

    # Load foundation object poses
    foundation_path = "foundation_pose.json"
    foundation_data = load_foundation_pose(foundation_path)

    # Example: define real2sim transform
    def real2sim_transform(real_pos):
        # identity for demo, but you could do scaling, flipping, etc.
        return real_pos

    # Edit MJCF
    original_mjcf = "scene.xml"
    output_mjcf = "scene_modified.xml"
    edit_mjcf_xml(
        xml_path=original_mjcf,
        output_path=output_mjcf,
        foundation_data=foundation_data,
        real2sim_transform=real2sim_transform
    )

if __name__ == "__main__":
    main()
