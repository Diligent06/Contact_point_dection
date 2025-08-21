import open3d as o3d
import pickle
from grasp_utils import create_radius


with open('./hit_points.pkl', 'rb') as f:
    data = pickle.load(f)

hit_points = data['hit_points']
hit_dis = data['hit_distances']

pcd = o3d.io.read_point_cloud("merged_point_cloud.pcd") 
contact_point_list = []
for i in range(len(hit_points)):
    contact_point = hit_points[i]
    contact_point_list.append(create_radius(contact_point, radius=0.01, color=[1, 0, 0]))

o3d.visualization.draw_geometries([pcd] + contact_point_list)