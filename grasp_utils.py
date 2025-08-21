import numpy as np
import open3d as o3d
def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
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

def plot_gripper_pro_max(center, R, width, depth, score=1, height=0.004, color=None):
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
  height=height
  finger_width = 0.004
  tail_length = 0.04
  depth_base = 0.02
  
  if color is not None:
      color_r, color_g, color_b = color
  else:
      color_r = score # red for high score
      color_g = 0
      color_b = 1 - score # blue for low score
  
  left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
  right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
  bottom = create_mesh_box(finger_width, width, height)
  tail = create_mesh_box(tail_length, finger_width, height)

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

def create_direction_arrow(translation, direction_vec, arrow_length=0.1, color=[1, 0, 0]):
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

def create_radius(center, radius=0.01, color=[0, 1, 0]):
    z_radius = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    z_radius.translate(center)
    z_radius.paint_uniform_color(color)  # green color
    return z_radius