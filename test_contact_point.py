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

# o3d.visualization.draw_geometries([merged_mesh, merged_pcd]) 

points = np.asarray(merged_pcd.points)

o3d.io.write_point_cloud("merged_point_cloud.pcd", merged_pcd)

translation = np.array([-0.00654207, -0.00167324, -0.05620303])

rotation = [[-0.01589504,  0.99617106,  0.08596864],
            [-0.18111572, -0.08742573,  0.97956824],
            [ 0.98333335,  0.,          0.18181187]]
rotation = np.array(rotation)

width = 0.08765953779220581
height = 0.019999999552965164
depth = 0.029999999329447746


arrow_x = create_direction_arrow(translation, rotation[:, 0], arrow_length=1, color=[1, 0, 0])
arrow_y = create_direction_arrow(translation, rotation[:, 1], arrow_length=1, color=[0, 1, 0])
arrow_z = create_direction_arrow(translation, rotation[:, 2], arrow_length=1, color=[0, 0, 1])


gripper = plot_gripper_pro_max(center=translation, R=rotation, width=width, depth=depth, score=0.8, color=(1, 0, 0))

scene = o3d.t.geometry.RaycastingScene()
mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(merged_mesh))

ee_base_point = translation - rotation[:, 0] * depth
finger_center_point = translation + rotation[:, 0] * depth
inter_len = 0.001
inter_num = int(depth * 2 / inter_len)
print(f'inter_num: {inter_num}')

hit_point_list_lr = []
hit_point_dis_list_lr = []

hit_point_list_rl = []
hit_point_dis_list_rl = []

for i in range(inter_num):
    cur_point = ee_base_point + rotation[:, 0] * 2 * depth * i / inter_num
    cur_point_l = cur_point + rotation[:, 1] * width / 2.0
    cur_point_r = cur_point - rotation[:, 1] * width / 2.0
    
    radius_l = create_radius(cur_point_l, radius=0.002, color=[0, 0, 1])
    radius_r = create_radius(cur_point_r, radius=0.002, color=[0, 0, 1])
    direction_lr = cur_point_r - cur_point_l
    direction_lr = direction_lr / np.linalg.norm(direction_lr)
    direction_rl = cur_point_l - cur_point_r
    direction_rl = direction_rl / np.linalg.norm(direction_rl)
    
    cur_point_l = cur_point_l.tolist()
    cur_point_r = cur_point_r.tolist()
    direction_rl = direction_rl.tolist()
    direction_lr = direction_lr.tolist()
    
    
    rays = o3d.core.Tensor([cur_point_l + direction_lr], dtype=o3d.core.Dtype.Float32) # 从l指向r
    ans = scene.cast_rays(rays)

    hit_dis = ans['t_hit'].numpy()
    hit_point = (rays[0,:3] + ans['t_hit'][0] * rays[0,3:6]).numpy()
    
    if hit_dis[0] != np.inf:
        hit_point_list_lr.append(hit_point)
        hit_point_dis_list_lr.append(hit_dis)
    else:
        hit_point_list_lr.append(None)
        hit_point_dis_list_lr.append(None)
        
    rays = o3d.core.Tensor([cur_point_r + direction_rl], dtype=o3d.core.Dtype.Float32) # 从l指向r
    ans = scene.cast_rays(rays)

    hit_dis = ans['t_hit'].numpy()
    hit_point = (rays[0,:3] + ans['t_hit'][0] * rays[0,3:6]).numpy()
    
    if hit_dis[0] != np.inf:        
        hit_point_list_rl.append(hit_point)
        hit_point_dis_list_rl.append(hit_dis)
    else:
        hit_point_list_rl.append(None)
        hit_point_dis_list_rl.append(None)

final_hit_point_pair_l = []
final_hit_point_pair_r = []
final_hit_dis_pair_l = []
final_hit_dis_pair_r = []

for i in range(len(hit_point_list_lr)):
    if hit_point_list_lr[i] is not None and hit_point_list_rl[i] is not None:
        final_hit_point_pair_l.append(hit_point_list_lr[i])
        final_hit_point_pair_r.append(hit_point_list_rl[i])
        final_hit_dis_pair_l.append(hit_point_dis_list_lr[i])
        final_hit_dis_pair_r.append(hit_point_dis_list_rl[i])

        
final_hit_point_pair_l = np.array(final_hit_point_pair_l)
final_hit_dis_pair_l = np.array(final_hit_dis_pair_l)

final_hit_point_pair_r = np.array(final_hit_point_pair_r)
final_hit_dis_pair_r = np.array(final_hit_dis_pair_r)

print(f'len final_hit_point_pair_l: {len(final_hit_point_pair_l)}, len final_hit_point_pair_r: {len(final_hit_point_pair_r)}')

with open("hit_points.pkl", "wb") as f:
    pickle.dump({"final_hit_point_pair_l": final_hit_point_pair_l, "final_hit_dis_pair_l": final_hit_dis_pair_l,
                 "final_hit_point_pair_r": final_hit_point_pair_r, "final_hit_dis_pair_r": final_hit_dis_pair_r}, f)


        







