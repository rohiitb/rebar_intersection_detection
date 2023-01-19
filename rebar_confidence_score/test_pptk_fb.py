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

point_cloud = vis_util.read_ply_as_PointCloud('data') # read the ply file in data/, should only have 1 ply file
image_dataset = database.ImgDataBase(root = root)
# point_cloud.add(vis_util.draw_allCameras(image_dataset))

# idx = 0; uv=[ 2723,2067]
# projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
# point_proj = np.mean(projection_point.xyz,axis=0)

# create a line 
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

points_proj_1 = []
for uv in uv_line_1:
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    point_proj = np.mean(projection_point.xyz,axis=0)
    points_proj_1.append(point_proj)

points_proj_2 = []
for uv in uv_line_2:
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    point_proj = np.mean(projection_point.xyz,axis=0)
    points_proj_2.append(point_proj)

points_proj_3 = []
for uv in uv_line_3:
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    point_proj = np.mean(projection_point.xyz,axis=0)
    points_proj_3.append(point_proj)

points_proj_4 = []
for uv in uv_line_4:
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    point_proj = np.mean(projection_point.xyz,axis=0)
    points_proj_4.append(point_proj)

points_proj_5 = []
for uv in uv_line_5:
    projection_point, projection_line, points_filtered = camera.projection_img2world_point(uv,idx,image_dataset,point_cloud.xyz)
    point_proj = np.mean(projection_point.xyz,axis=0)
    points_proj_5.append(point_proj)


for idx in range(len(image_dataset)):
    us_projected_1 = []
    vs_projected_1 = []
    for point_proj in points_proj_1:
        uv_projected, if_inside_1 = camera.projection_world2img(point_proj,idx,image_dataset)
        if if_inside_1:
            us_projected_1.append(uv_projected[0])
            vs_projected_1.append(uv_projected[1])

    us_projected_2 = []
    vs_projected_2 = []
    for point_proj in points_proj_2:
        uv_projected, if_inside_2 = camera.projection_world2img(point_proj,idx,image_dataset)
        if if_inside_2:
            us_projected_2.append(uv_projected[0])
            vs_projected_2.append(uv_projected[1])

    us_projected_3 = []
    vs_projected_3 = []
    for point_proj in points_proj_3:
        uv_projected, if_inside_3 = camera.projection_world2img(point_proj,idx,image_dataset)
        if if_inside_3:
            us_projected_3.append(uv_projected[0])
            vs_projected_3.append(uv_projected[1])

    us_projected_4 = []
    vs_projected_4 = []
    for point_proj in points_proj_4:
        uv_projected, if_inside_4 = camera.projection_world2img(point_proj,idx,image_dataset)
        if if_inside_4:
            us_projected_4.append(uv_projected[0])
            vs_projected_4.append(uv_projected[1])

    us_projected_5 = []
    vs_projected_5 = []
    for point_proj in points_proj_5:
        uv_projected, if_inside_5 = camera.projection_world2img(point_proj,idx,image_dataset)
        if if_inside_5:
            us_projected_5.append(uv_projected[0])
            vs_projected_5.append(uv_projected[1])

    axs = plt.subplot(111)
    im = image_dataset[idx]
    axs.imshow(im)
    # axs.set_title(if_inside)
        # axs.scatter(us_projected,vs_projected)
    axs.plot(us_projected_1,vs_projected_1, linestyle='--', marker='o', color='b',
    linewidth=0.8, markersize=1.2,label='rebar')
    axs.plot(us_projected_2, vs_projected_2, linestyle='--', marker='o', color='b',
                linewidth=0.8, markersize=1.2, label='rebar')
    axs.plot(us_projected_3, vs_projected_3, linestyle='--', marker='o', color='b',
                linewidth=0.8, markersize=1.2, label='rebar')
    axs.plot(us_projected_4, vs_projected_4, linestyle='--', marker='o', color='b',
                linewidth=0.8, markersize=1.2, label='rebar')
    axs.plot(us_projected_5, vs_projected_5, linestyle='--', marker='o', color='b',
                linewidth=0.8, markersize=1.2, label='rebar')

    plt.show()