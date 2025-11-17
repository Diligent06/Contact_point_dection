import numpy as np
import open3d as o3d
import open3d as o3d
import numpy as np

import os
from os.path import join
import sys
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'contact_grasp_cal'))

import pickle


class ContactDetector:
    def __init__(self, iter_len=0.001):
        self.inter_len = iter_len
        pass
    
    def find_lr_contact(self, obj_mesh, grasp_config):
        rotation = grasp_config['rotation']
        width = grasp_config['width']
        depth = grasp_config['depth']
        ee_base_point = grasp_config['ee_base_point']
        
        approach_vec = grasp_config['approach']
        binormal_vec = grasp_config['binormal']
        
        scene = o3d.t.geometry.RaycastingScene()
        mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj_mesh))

        inter_len = 0.001
        inter_num = int(depth * 2 / inter_len)
        print(f'inter_num: {inter_num}')

        hit_point_list_lr = []
        hit_point_dis_list_lr = []

        hit_point_list_rl = []
        hit_point_dis_list_rl = []

        for i in range(inter_num):
            cur_point = ee_base_point + approach_vec * depth * i / inter_num
            cur_point_l = cur_point + binormal_vec * width / 2.0
            cur_point_r = cur_point - binormal_vec * width / 2.0
            
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
        
        return final_hit_point_pair_l, final_hit_dis_pair_l, final_hit_point_pair_r, final_hit_dis_pair_r

    def find_hit_point(self, final_hit_point_pair_l, final_hit_dis_pair_l, final_hit_point_pair_r, final_hit_dis_pair_r):
        contact_candidate_point_list = []
        for i in range(len(final_hit_point_pair_l)):
            contact_point = final_hit_point_pair_l[i]
            contact_candidate_point_list.append(self.create_radius(contact_point, radius=0.002, color=[1, 0, 0]))
        for i in range(len(final_hit_point_pair_r)):
            contact_point = final_hit_point_pair_r[i]
            contact_candidate_point_list.append(self.create_radius(contact_point, radius=0.002, color=[0, 1, 0]))

        dis = 1000
        contact_id = -1
        for i in range(len(final_hit_point_pair_l)):
            cur_dis = final_hit_dis_pair_l[i] + final_hit_dis_pair_r[i]
            if cur_dis < dis:
                dis = cur_dis
                contact_id = i

        final_contact_point_l = final_hit_point_pair_l[contact_id] if contact_id != -1 else None
        final_contact_point_r = final_hit_point_pair_r[contact_id] if contact_id != -1 else None
        
        return final_contact_point_l, final_contact_point_r

    def visualize_contact_point(self, pcd, contact_point_l, contact_point_r, grasp_config):
        contact_candidate_point_list = []
        if contact_point_l is not None:
            contact_candidate_point_list.append(self.create_radius(contact_point_l, radius=0.01, color=[0, 0, 1]))
        if contact_point_r is not None:
            contact_candidate_point_list.append(self.create_radius(contact_point_r, radius=0.01, color=[0, 0, 1]))
        
        translation = grasp_config['translation']
        rotation = grasp_config['rotation']
        width = grasp_config['width']
        depth = grasp_config['depth']
        gripper = self.plot_gripper_pro_max(center=translation, R=rotation, width=width, depth=depth, score=0.8, color=(1, 0, 0))
        o3d.visualization.draw_geometries([pcd, gripper] + contact_candidate_point_list)
    
    def create_mesh_box(self, width, height, depth, dx=0, dy=0, dz=0):
        ''' Author: chenxi-wang
        Create box instance with mesh representation.
        '''
        box = o3d.geometry.TriangleMesh()
        vertices = np.array([[0,0,0],
                                [width,0,0],
                                [0,0,depth],
                                [width,0,depth],
                                [0,height,0],
                                [width,height,0],
                                [0,height,depth],
                                [width,height,depth]])
        vertices[:,0] += dx
        vertices[:,1] += dy
        vertices[:,2] += dz
        triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                                [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                                [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
        box.vertices = o3d.utility.Vector3dVector(vertices)
        box.triangles = o3d.utility.Vector3iVector(triangles)
        return box

    def plot_gripper_pro_max(self, center, R, width, depth, score=1, height=0.004, color=None):
        '''
        Author: chenxi-wang
        
        **Input:**

        - center: numpy array of (3,), target point as gripper center

        - R: numpy array of (3,3), rotation matrix of gripper

        - width: float, gripper width

        - score: float, grasp quality score

        **Output:**

        - open3d.geometry.TriangleMesh
        '''
        x, y, z = center
        height = height
        width = 0.04 if width is None else width
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02 
        
        if color is not None:
            color_r, color_g, color_b = color
        else:
            color_r = score # red for high score
            color_g = 0
            color_b = 1 - score # blue for low score
        
        left = self.create_mesh_box(depth+depth_base+finger_width, finger_width, height)
        right = self.create_mesh_box(depth+depth_base+finger_width, finger_width, height)
        bottom = self.create_mesh_box(finger_width, width, height)
        tail = self.create_mesh_box(tail_length, finger_width, height)

        left_points = np.array(left.vertices)
        left_triangles = np.array(left.triangles)
        left_points[:,0] -= depth_base + finger_width
        left_points[:,1] -= width/2 + finger_width
        left_points[:,2] -= height/2

        right_points = np.array(right.vertices)
        right_triangles = np.array(right.triangles) + 8
        right_points[:,0] -= depth_base + finger_width
        right_points[:,1] += width/2
        right_points[:,2] -= height/2

        bottom_points = np.array(bottom.vertices)
        bottom_triangles = np.array(bottom.triangles) + 16
        bottom_points[:,0] -= finger_width + depth_base
        bottom_points[:,1] -= width/2
        bottom_points[:,2] -= height/2

        tail_points = np.array(tail.vertices)
        tail_triangles = np.array(tail.triangles) + 24
        tail_points[:,0] -= tail_length + finger_width + depth_base
        tail_points[:,1] -= finger_width / 2
        tail_points[:,2] -= height/2

        vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
        vertices = np.dot(R, vertices.T).T + center
        triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
        colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

        gripper = o3d.geometry.TriangleMesh()
        gripper.vertices = o3d.utility.Vector3dVector(vertices)
        gripper.triangles = o3d.utility.Vector3iVector(triangles)
        gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
        return gripper

    def create_axis_open3d(self, size=0.1, position=[1, 0, 0]):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,   # 坐标轴长度
            origin=position  # 原点位置
        )
        return axis

    def create_radius(self, center, radius=0.01, color=[0, 1, 0]):
        z_radius = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        z_radius.translate(center)
        z_radius.paint_uniform_color(color)  # green color
        return z_radius

    def create_direction_arrow(self, translation, direction_vec, arrow_length=0.1, color=[1, 0, 0]):
        arrow_direction = direction_vec# * arrow_length

        # Open3D's create_arrow creates an arrow along +Z, so we need to align it
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.003, cone_radius=0.006,
            cylinder_height=arrow_length * 0.8, cone_height=arrow_length * 0.2
        )
        arrow.paint_uniform_color(color)  # red color

        # Compute rotation matrix to align +Z to arrow_direction
        z = np.array([0, 0, 1])
        v = np.cross(z, arrow_direction)
        c = np.dot(z, arrow_direction)
        if np.linalg.norm(v) < 1e-8:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
        arrow.rotate(R, center=np.zeros(3))
        arrow.translate(translation)
        
        return arrow
        
    
    
    
    