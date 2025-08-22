import open3d as o3d
import pickle
from grasp_utils import create_radius, plot_gripper_pro_max
import numpy as np

with open('./hit_points.pkl', 'rb') as f:
    data = pickle.load(f)

final_hit_point_pair_l = data['final_hit_point_pair_l']
final_hit_dis_pair_l = data['final_hit_dis_pair_l']
final_hit_point_pair_r = data['final_hit_point_pair_r']
final_hit_dis_pair_r = data['final_hit_dis_pair_r']


contact_candidate_point_list = []
for i in range(len(final_hit_point_pair_l)):
    contact_point = final_hit_point_pair_l[i]
    contact_candidate_point_list.append(create_radius(contact_point, radius=0.002, color=[1, 0, 0]))
for i in range(len(final_hit_point_pair_r)):
    contact_point = final_hit_point_pair_r[i]
    contact_candidate_point_list.append(create_radius(contact_point, radius=0.002, color=[0, 1, 0]))

dis = 1000
contact_id = -1
for i in range(len(final_hit_point_pair_l)):
    cur_dis = final_hit_dis_pair_l[i] + final_hit_dis_pair_r[i]
    if cur_dis < dis:
        dis = cur_dis
        contact_id = i

final_contact_point_l = final_hit_point_pair_l[contact_id] if contact_id != -1 else None
final_contact_point_r = final_hit_point_pair_r[contact_id] if contact_id != -1 else None

final_contact_radius_l = create_radius(final_contact_point_l, radius=0.01, color=[0, 0, 1]) if contact_id != -1 else None
final_contact_radius_r = create_radius(final_contact_point_r, radius=0.01, color=[0, 0, 1]) if contact_id != -1 else None

_ = [contact_candidate_point_list.append(final_contact_radius_l) if final_contact_radius_l is not None else None]
_ = [contact_candidate_point_list.append(final_contact_radius_r) if final_contact_radius_r is not None else None]

# from IPython import embed; embed()

pcd = o3d.io.read_point_cloud("merged_point_cloud.pcd") 

rotation = [[-0.01589504,  0.99617106,  0.08596864],
            [-0.18111572, -0.08742573,  0.97956824],
            [ 0.98333335,  0.,          0.18181187]]
rotation = np.array(rotation)
translation = np.array([-0.00654207, -0.00167324, -0.05620303])

width = 0.08765953779220581
height = 0.019999999552965164
depth = 0.029999999329447746


gripper = plot_gripper_pro_max(center=translation, R=rotation, width=width, depth=depth, score=0.8, color=(1, 0, 0))

# left_point = translation + depth * rotation[:, 0] + width * rotation[:, 1] / 2
# left_point_radius = create_radius(left_point, radius=0.01, color=[1, 0, 0])
# right_point = translation + depth * rotation[:, 0] - width * rotation[:, 1] / 2
# right_point_radius = create_radius(right_point, radius=0.01, color=[1, 0, 0])
# contact_candidate_point_list.append(left_point_radius)
# contact_candidate_point_list.append(right_point_radius)

o3d.visualization.draw_geometries([pcd, gripper] + contact_candidate_point_list)

points = np.asarray(pcd.points)
# Calculate distances from all points to final_contact_point_l
print("Calculating distances from all points to final_contact_point_l")
if final_contact_point_l is not None:
    distances = np.linalg.norm(points - final_contact_point_l, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_point = points[min_idx]
    print(f"Minimal distance: {min_distance}")
    print(f"Closest point: {min_point}")
else:
    print("No valid final_contact_point_l found.")

print("Calculating distances from all points to final_contact_point_r")
if final_contact_point_r is not None:
    distances = np.linalg.norm(points - final_contact_point_r, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_point = points[min_idx]
    print(f"Minimal distance: {min_distance}")
    print(f"Closest point: {min_point}")
else:
    print("No valid final_contact_point_r found.")

