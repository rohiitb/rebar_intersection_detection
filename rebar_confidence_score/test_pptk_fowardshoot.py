import pptk
import numpy as np
import plyfile
import matplotlib.pyplot as plt
from src.vision import database, camera
from src.vision.util import str2double
import glob
from sklearn.neighbors import NearestNeighbors
import src.visual3d.util as vis_util
from src.visual3d import geometry

# data directory
root = "data"

# read 3D reconstruction point cloud
point_cloud = vis_util.read_ply_as_PointCloud('data') # read the ply file in data/, should only have 1 ply file

# load image and camera dataset
image_dataset = database.ImgDataBase(root = root)

# add all cameras to visualization
point_cloud.add(vis_util.draw_allCameras(image_dataset))

# create projection proints from image
idx = 0
uv_start_1 = [292,1901];uv_end_1 = [3817, 2158]
num_pointOfLine = 5
u_line_1 = np.linspace(uv_start_1[0],uv_end_1[0],num=num_pointOfLine)
v_line_1 = np.linspace(uv_start_1[1],uv_end_1[1],num=num_pointOfLine)
uv_line_1 = np.c_[u_line_1,v_line_1].astype('int')

uv_start_2 = [11,2029];uv_end_2 = [3789, 2388]
num_pointOfLine = 5
u_line_2 = np.linspace(uv_start_2[0],uv_end_2[0],num=num_pointOfLine)
v_line_2 = np.linspace(uv_start_2[1],uv_end_2[1],num=num_pointOfLine)
uv_line_2 = np.c_[u_line_2,v_line_2].astype('int')

uv_line_3 = np.array([[2734, 2354],[2228,2300],[1753,2257],[1257,2202],[903,2167],[748,2148]])
uv_line_4 = np.array([[1906, 534],[1555,1059],[1247,1523],[932,1992],[713,2311],[1729,795]])
uv_line_5 = np.array([[2014, 562],[1810,900],[1568,1290],[1319,1703],[1120,2036],[904,2385]])

# compute the projection point from image to real world
projection_iter = 0
projection_geoms = []
for uv in uv_line_1:
    # compute projection line from camera to 3D world
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    projection_geoms.append([projection_point,projection_line,points_filtered])
    projection_iter +=1
    print(f"completed the {projection_iter}th projection")

for uv in uv_line_2:
    # compute projection line from camera to 3D world
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    projection_geoms.append([projection_point,projection_line,points_filtered])
    projection_iter +=1
    print(f"completed the {projection_iter}th projection")

for uv in uv_line_3:
    # compute projection line from camera to 3D world
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    projection_geoms.append([projection_point,projection_line,points_filtered])
    projection_iter +=1
    print(f"completed the {projection_iter}th projection")

for uv in uv_line_4:
    # compute projection line from camera to 3D world
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    projection_geoms.append([projection_point,projection_line,points_filtered])
    projection_iter +=1
    print(f"completed the {projection_iter}th projection")

for uv in uv_line_5:
    # compute projection line from camera to 3D world
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    projection_geoms.append([projection_point,projection_line,points_filtered])
    projection_iter +=1
    print(f"completed the {projection_iter}th projection")


# add the projection geometries
for projection_point, projection_line, points_filtered in projection_geoms:
    point_cloud.add(projection_point,projection_line,points_filtered)

# uv=[ 2723,2067]
# projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)

# add the projection line and point to visualization
# point_cloud.add(projection_point,projection_line,points_filtered)

# visualize the result
v = vis_util.visualize_pointCloud(point_cloud)

# print out the 3D postion of the projection
point_proj = np.mean(projection_point.xyz,axis=0)
print(f"the world projected position is: {point_proj}")
