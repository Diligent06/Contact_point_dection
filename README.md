# 1. Visualize gripper mesh in open3d
utils/visualize_gripper.py

create_gripper_geometry() can input transformation of gripper and output a mesh which can be visualize by open3d. (o3d.visualization.draw_geometries(gripper))

# 2. Find contact points from a two-finger gripper
Find contact points of a two-finger gripper closure action. Input should contain object mesh and gripper infomation. (gripper coordinate definition is the same with https://github.com/atenpas/gpd)

Find contact point demo: test_contact_point.py (contact_detector contains a class which can be used easily); hit_points.pkl is the result of contact points and merged_point_cloud.pcd is merged pointcloud from 8554 objects. Finally visualize_pcd.py can visualize merged_point_cloud and contact points.