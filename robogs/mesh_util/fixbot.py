import numpy as np
import trimesh

mesh = trimesh.load("A.stl")


threshold = 0.001
# 2. Compute bounding box along the y-axis
min_z = mesh.bounds[0][1]
max_z = mesh.bounds[1][1]

height = max_z - min_z
print(f"min_z: {min_z}, max_z: {max_z}, height: {height}")
# 3. Threshold at 95% above the min_y
z_threshold = min_z + threshold * (max_z - min_z)

# 4. Keep only vertices above that threshold
vertex_mask = mesh.vertices[:, 2] >= -0.069

# 5. Keep only faces for which *all three* vertices pass the mask
faces_mask = vertex_mask[mesh.faces].all(axis=1)

# 6. Create a new mesh with the filtered geometry
submesh = mesh.copy()
submesh.update_faces(faces_mask)
submesh.remove_unreferenced_vertices()

# 7. Export the submesh to a new file
submesh.export("A_clean_new.stl")