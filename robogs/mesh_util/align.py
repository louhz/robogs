# icp
import open3d as o3d
import numpy as np

def load_mesh_or_pcd(file_path):
    if file_path.endswith(".ply") or file_path.endswith(".pcd") or file_path.endswith(".xyz"):
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
    return pcd

def run_icp(source_pcd, target_pcd, voxel_size=0.005, max_iter=50, threshold=0.02):
    # Downsample
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Initial transformation (identity)
    trans_init = np.eye(4)

    # Run ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    print("Transformation Matrix:\n", reg_p2p.transformation)
    return reg_p2p.transformation, reg_p2p.inlier_rmse

def visualize_alignment(source_pcd, target_pcd, transformation):
    source_temp = source_pcd.transform(transformation)
    o3d.visualization.draw_geometries([source_temp.paint_uniform_color([1, 0, 0]),
                                       target_pcd.paint_uniform_color([0, 1, 0])])

# Example Usage
if __name__ == "__main__":
    source_file = "source_mesh_or_pcd.ply"
    target_file = "target_mesh_or_pcd.ply"

    source = load_mesh_or_pcd(source_file)
    target = load_mesh_or_pcd(target_file)

    transformation, rmse = run_icp(source, target)
    print(f"Final RMSE: {rmse:.6f}")

    visualize_alignment(source, target, transformation)