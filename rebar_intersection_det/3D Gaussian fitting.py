import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
from sklearn import mixture
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import scipy
import src.realsense.camera as rlc
import pyransac3d as pyrsc
import rosbag
import src.visual3d.util as vis_util
from src.visual3d import geometry
import time
import open3d as o3d
from sklearn.decomposition import PCA

def line_fit(points):
    pca = PCA(n_components=1)
    pca.fit(points)
    dir_vec = pca.components_
    mean = np.mean(points, axis=0)
    linepts = dir_vec * np.mgrid[-0.5:0.5:2j][:, np.newaxis]
    linepts += mean
    return linepts[0], linepts[-1]

def visualize_3d_gmm(points, points_rgb, w, mu, stdev):
    n_gaussians = mu.shape[1]

    for i in range(n_gaussians):

        stacked_p = geometry.Ellipsoid(position=mu[:, i], radius=stdev[:, i]).compute_xyz()
        points = np.vstack((points, stacked_p))

        stacked_rgb = geometry.Ellipsoid(position=mu[:, i], radius=stdev[:, i]).compute_rgb()
        points_rgb = np.vstack((points_rgb, stacked_rgb))

    v = vis_util.visualize_pointCloud_in_xyz(point_cloud=points, rgb=points_rgb)

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("data/single_layer.bag")
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(1000):
    pipe.wait_for_frames()

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
print(f"shape of rgb = {color.shape}")

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
print(f"shape of colorized_depth = {colorized_depth.shape}")
H,W = color.shape[:2]

gray = np.mean(colorized_depth,axis=2).astype(np.uint8)
plt.imshow(color)

align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())


# Show the two frames together:
images = np.hstack((color, colorized_depth))
plt.imshow(images)
# plt.imshow(colorized_depth_cp)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
# print(depth_scale)

depth = np.asanyarray(aligned_depth_frame.get_data())
depth = depth*depth_scale
print(f"shape of colorized_depth = {colorized_depth.shape}")

bag = rosbag.Bag('data/single_layer.bag')
count_loop = 0
for topic, msg, t in bag.read_messages():
    if 'Color_0/info/camera_info' in topic:
        K = msg.K
    count_loop +=1
    if count_loop > 20:
        break
bag.close()

K = np.array(K).reshape(3,3)
print(f"Obtained the intrinsics: K= \n{K}")

from sklearn import linear_model
xyz,rgb = rlc.project_rgbd2pcd(color,depth,K) # project to point cloud

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb/255.)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])

idx = xyz[:,2]>0.001
xyz = xyz[idx,:]; rgb = rgb[idx,:]

xy = xyz[:,:2]; z=xyz[:,2]
reg = linear_model.LinearRegression()
reg.fit(xy,z)

# print(f"plane: z = {reg.coef_[0]}x + {reg.coef_[1]}y + {reg.intercept_} ")
class_binary = (z - (xy @ reg.coef_ + reg.intercept_) ) < -0.020                   # Previously -0.030

xyz_top = xyz[class_binary,:]
rgb_top = rgb[class_binary,:]
xyz_bottom = xyz[~class_binary,:]
rgb_bottom = rgb[~class_binary,:]


# point_cloud = geometry.PointCloud(xyz_top)
# point_cloud.rgb = rgb_top
t0 = time.time()

needed_gaussians = 7
gmm = mixture.BayesianGaussianMixture(n_components=needed_gaussians, covariance_type='full')
gmm.fit(xyz_top)
t1 = time.time()

total = t1-t0
print("Time : ", total)

Y_ = gmm.predict(xyz_top)
cloud = geometry.PointCloud(xyz=xyz_top, color=rgb_top)
# cloud.show()

line_3d = pyrsc.Line()
t2 = time.time()
cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 0, :], color='red'))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 0, :], thresh=0.5, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 0, :][st], end=xyz_top[Y_ == 0, :][en], color='white', linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 0, :])[0], end=line_fit(xyz_top[Y_ == 0, :])[1], color='white', linewidth=0.003))
t3 = time.time()
print("Time for one line fitting : ", t3-t2)

t4 = time.time()
cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 1, :], color='green'))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 1, :], thresh=0.2, maxIteration=1000)      # Using RANSAC
# st = int(inliers[0]) ; en = int(inliers[-1])

cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 1, :])[0], end=line_fit(xyz_top[Y_ == 1, :])[1], color='white', linewidth=0.003))
t5 = time.time()
print("Time for second line fitting : ", t5- t4)

cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 2, :], color='blue'))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 2, :], thresh=0.5, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 2, :][st], end=xyz_top[Y_ == 2, :][en], color='white', linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 2, :])[0], end=line_fit(xyz_top[Y_ == 2, :])[1], color='white', linewidth=0.003))


cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 3, :], color=[211,84,80]))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 3, :], thresh=0.5, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 3, :][st], end=xyz_top[Y_ == 3, :][en], color='white', linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 3, :])[0], end=line_fit(xyz_top[Y_ == 3, :])[1], color='white', linewidth=0.003))


cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 4, :], color=[165,105,189]))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 4, :], thresh=0.5, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 4, :][st], end=xyz_top[Y_ == 4, :][en], color='white',linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 4, :])[0], end=line_fit(xyz_top[Y_ == 4, :])[1], color='white', linewidth=0.003))


cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 5, :], color=[165,105,0]))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 5, :], thresh=0.2, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 5, :][st], end=xyz_top[Y_ == 5, :][en], color='white',linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 5, :])[0], end=line_fit(xyz_top[Y_ == 5, :])[1], color='white', linewidth=0.003))
#
cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 6, :], color=[256,84,10]))
# slope, line_intercept, inliers = line_3d.fit(xyz_top[Y_ == 6, :], thresh=0.2, maxIteration=1000)
# st = int(inliers[0]) ; en = int(inliers[-1])
# cloud.add(geometry.Line(start=xyz_top[Y_ == 6, :][st], end=xyz_top[Y_ == 6, :][en], color='white',linewidth=0.003))
cloud.add(geometry.Line(start=line_fit(xyz_top[Y_ == 6, :])[0], end=line_fit(xyz_top[Y_ == 6, :])[1], color='white', linewidth=0.003))

# cloud.add(geometry.PointCloud(xyz=xyz_top[Y_== 4, :], color=[165,70,189]))
#
cloud.show()


# cov_mat = np.zeros((needed_gaussians, 3))


# for i in range(needed_gaussians):
#     eigval, eigvec = scipy.linalg.eigh(gmm.covariances_[i, :, :])
#     cov_mat[i] = eigvec[0]
#
#     print(eigvec[:,1])
# print("Cov_mat shape : ", cov_mat)


# visualize_3d_gmm(xyz_top,rgb_top, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
# visualize_3d_gmm(xyz_top,rgb_top, gmm.weights_, gmm.means_.T, cov_mat.T)
# print("Covariances : ", gmm.covariances_.shape)
# print("Covariances : ", gmm.covariances_.shape)
# print("Mean : ", gmm.means_)
# print("Weights : ", gmm.weights_)