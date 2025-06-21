import openai
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import json
from PIL import Image
import base64


OPENAI_API_KEY = 'YOUR_API_KEY_HERE'

client = openai.OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_with_gpt(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content


def compute_mesh_properties(raw_image_path, seg_image_path):
    prompt = (
        "You are a helpful robotics and physics assistant. Given two images, one raw RGB and one segmentation, please infer "
        "the object's mass and friction. Return a JSON with the keys 'mass' and 'friction'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(raw_image_path)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(seg_image_path)}"}}
            ]
        }
    ]

    response = chat_with_gpt(messages)

    try:
        properties = json.loads(response)
        return properties['mass'], properties['friction']
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON received from OpenAI.")


def update_mjcf(scene_path, output_path, ply_path, raw_image_path, seg_image_path):
    mass, friction = compute_mesh_properties(raw_image_path, seg_image_path)

    mesh = trimesh.load(ply_path)
    com = mesh.bounding_box.centroid

    # Translate mesh to origin based on COM
    mesh.apply_translation(-com)

    tree = ET.parse(scene_path)
    root = tree.getroot()

    cup_body = root.find(".//body[@name='cup']")
    geom = cup_body.find("geom[@name='cup']")

    geom.set('mass', f"{mass}")
    geom.set('friction', ' '.join(map(str, friction)))

    cup_body.set('pos', ' '.join(map(str, com)))

    prompt_name = "Provide a creative name for an object to be used in a robotic grasping simulation."
    response_name = chat_with_gpt([{"role": "user", "content": prompt_name}])

    new_name = response_name.strip().replace(' ', '_').lower()
    cup_body.set('name', new_name)
    geom.set('name', new_name)

    tree.write(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='Path to output MJCF file')
    parser.add_argument('-s', '--seed', required=True, help='Path to input seed PLY file')
    parser.add_argument('-m', '--mjcf', default='scene.xml', help='Path to original MJCF scene file')
    parser.add_argument('--raw_image', required=True, help='Path to raw RGB image')
    parser.add_argument('--seg_image', required=True, help='Path to segmentation image')

    args = parser.parse_args()

    update_mjcf(args.mjcf, args.output, args.seed, args.raw_image, args.seg_image)


# note that you may need to fix the scale of the object centor of mass for minimum sim2real gap