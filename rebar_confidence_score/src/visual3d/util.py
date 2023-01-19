from src.visual3d import geometry
import glob
import plyfile
import numpy as np
import pptk

def read_ply_as_PointCloud(root):
    path = glob.glob(root+'/*.ply')
    cloud_data = plyfile.PlyData.read(path[0])['vertex']
    point_cloud = geometry.PointCloud( np.c_[cloud_data['x'], cloud_data['y'], cloud_data['z']])
    point_cloud.rgb = np.c_[cloud_data['diffuse_red'], cloud_data['diffuse_green'], cloud_data['diffuse_blue']]
    return point_cloud

def draw_allCameras(image_dataset):
    point_cloud = geometry.PointCloud()
    for i in range(len(image_dataset)):
        camera_params = image_dataset.get_camera_attr(i,'all')
        point_cloud.add(geometry.Sphere(position=camera_params.t,radius=0.05,color='blue',num=2))
    return point_cloud

def visualize_pointCloud(point_cloud,lookat = [0,0,-2]):
    v = pptk.viewer(point_cloud.xyz)
    v.attributes(point_cloud.rgb / 255.)
    v.set(lookat=lookat)
    return v
