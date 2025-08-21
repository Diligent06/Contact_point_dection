import open3d as o3d
import numpy as np

from grasp_utils import plot_gripper_pro_max, create_direction_arrow, create_radius
import pickle

pcd_paths = ['./8554/objs/original-1.obj', './8554/objs/original-2.obj']

# Merge all point clouds into one
merged_pcd = o3d.geometry.PointCloud()
    
# Merge all meshes into one mesh
merged_mesh = o3d.geometry.TriangleMesh()
for pcd_file in pcd_paths:
    mesh = o3d.io.read_triangle_mesh(pcd_file)
    scale_factor = 0.1  # Set your desired scale factor here
    mesh.scale(scale_factor, center=[0, 0, 0])
    
    surface_area = mesh.get_surface_area()
    num_points = max(5000, int(surface_area * 500))

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    # Sample points from mesh to create a point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    # o3d.visualization.draw_geometries([pcd])
    
    merged_pcd += pcd

    merged_mesh += mesh  # Merge mesh

# Optionally, you can remove duplicated vertices and triangles
merged_mesh.remove_duplicated_vertices()
merged_mesh.remove_duplicated_triangles()
merged_mesh.remove_unreferenced_vertices()
merged_mesh.remove_degenerate_triangles()


points = np.asarray(merged_pcd.points)

o3d.io.write_point_cloud("merged_point_cloud.pcd", merged_pcd)

translation = np.array([-0.00654207, -0.00167324, -0.05620303])

rotation = [[-0.01589504,  0.99617106,  0.08596864],
            [-0.18111572, -0.08742573,  0.97956824],
            [ 0.98333335,  0.,          0.18181187]]



rotation = np.array(rotation)

x_axis = rotation[:, 0] + translation
y_axis = rotation[:, 1] + translation
z_axis = rotation[:, 2] + translation

x_radius = create_radius(x_axis)
y_radius = create_radius(y_axis)
z_radius = create_radius(z_axis)

arrow_x = create_direction_arrow(translation, rotation[:, 0], arrow_length=1, color=[1, 0, 0])
arrow_y = create_direction_arrow(translation, rotation[:, 1], arrow_length=1, color=[0, 1, 0])
arrow_z = create_direction_arrow(translation, rotation[:, 2], arrow_length=1, color=[0, 0, 1])


# Plot a small sphere at the translation point
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
sphere.translate(translation)
sphere.paint_uniform_color([0, 1, 0])  # green color

width = 0.08765953779220581
height = 0.019999999552965164
depth = 0.029999999329447746

gripper = plot_gripper_pro_max(center=translation, R=rotation, width=width, depth=depth, score=0.8, color=(1, 0, 0))

scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh))

y_axis_t = rotation[:, 1] + translation
y_axis_f = -rotation[:, 1] + translation

hit_point_list = []
hit_point_dis_list = []

for i in range(10):
    y_axis_t += rotation[:, 0] * 0.01
    y_axis_f += rotation[:, 0] * 0.01
    y_axis_t = y_axis_t.tolist()
    y_axis_f = y_axis_f.tolist()

    # 射线 (origin, direction)
    rays = o3d.core.Tensor([ y_axis_t + y_axis_f ], dtype=o3d.core.Dtype.Float32)  # 从(0,0,0) 朝x方向
    ans = scene.cast_rays(rays)

    hit_dis = ans['t_hit'].numpy()
    hit_point = (rays[0,:3] + ans['t_hit'][0] * rays[0,3:6]).numpy()
    
    if hit_dis[0] != np.inf:
        print(f'iter: {i + 10}')
        print("Hit distance:", hit_dis)
        print("Hit point:", hit_point)
        show_radius = create_radius(hit_point, radius=0.01, color=[1, 0, 0])
        # o3d.visualization.draw_geometries([merged_mesh, show_radius])
        
        hit_point_list.append(hit_point)
        hit_point_dis_list.append(hit_dis)

for i in range(10):
    y_axis_t -= rotation[:, 0] * 0.01
    y_axis_f -= rotation[:, 0] * 0.01
    y_axis_t = y_axis_t.tolist()
    y_axis_f = y_axis_f.tolist()

    # 射线 (origin, direction)
    rays = o3d.core.Tensor([ y_axis_t + y_axis_f ], dtype=o3d.core.Dtype.Float32)  # 从(0,0,0) 朝x方向
    ans = scene.cast_rays(rays)

    hit_dis = ans['t_hit'].numpy()
    hit_point = (rays[0,:3] + ans['t_hit'][0] * rays[0,3:6]).numpy()
    
    if hit_dis[0] != np.inf:
        print(f'iter: {i + 10}')
        print("Hit distance:", hit_dis)
        print("Hit point:", hit_point)
        show_radius = create_radius(hit_point, radius=0.01, color=[1, 0, 0])
        # o3d.visualization.draw_geometries([merged_mesh, show_radius])
        hit_point_list.append(hit_point)
        hit_point_dis_list.append(hit_dis)
        
hit_point_np = np.array(hit_point_list)
hit_dis_np = np.array(hit_point_dis_list)

with open("hit_points.pkl", "wb") as f:
    pickle.dump({"hit_points": hit_point_np, "hit_distances": hit_dis_np}, f)


        







