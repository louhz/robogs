import argparse
import numpy as np
import trimesh

def clean_mesh(input_path, output_path, threshold=0.01, z_cutoff=-0.069):
    mesh = trimesh.load(input_path)

    min_z = mesh.bounds[0][1]
    max_z = mesh.bounds[1][1]

    height = max_z - min_z
    print(f"min_z: {min_z}, max_z: {max_z}, height: {height}")

    z_threshold = min_z + threshold * (max_z - min_z)

    vertex_mask = mesh.vertices[:, 2] >= z_threshold

    faces_mask = vertex_mask[mesh.faces].all(axis=1)

    submesh = mesh.copy()
    submesh.update_faces(faces_mask)
    submesh.remove_unreferenced_vertices()

    submesh.export(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a mesh by removing vertices below a certain threshold.")

    parser.add_argument("-i", "--input", required=True, type=str, help="Path to input mesh file (.stl)")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to output cleaned mesh file (.stl)")
    parser.add_argument("--threshold", type=float, default=0.001, help="Threshold for bounding box calculation")
    parser.add_argument("--z_cutoff", type=float, default=-0.069, help="Z-axis cutoff to filter vertices")

    args = parser.parse_args()

    clean_mesh(args.input, args.output, args.threshold, args.z_cutoff)
