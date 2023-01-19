import numpy as np
import src.visual3d.util as vis_util
import pptk
from src.visual3d import geometry
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn import mixture


def points_ellipsoid(position, radius, num):
    phi = np.linspace(0, np.pi * 2, num)
    phi = np.expand_dims(phi, axis=0)
    th = np.linspace(0, np.pi * 2, num)
    th = np.expand_dims(th, axis=0)
    x = (radius[0] * (np.cos(phi.T) @ np.sin(th)) + position[0]).reshape(-1)
    y = (radius[1] * (np.sin(phi.T) @ np.sin(th)) + position[1]).reshape(-1)
    z = (radius[2] * (np.ones((num, 1)) @ np.cos(th)) + position[2]).reshape(-1)
    print("x: ", x.shape)
    print("y: ", y.shape)
    print("z: ", z.shape)
    xyz = np.c_[x, y, z]
    return xyz

def visualize_3d_gmm(points, points_rgb, w, mu, stdev, export=True):
    n_gaussians = mu.shape[1]


    for i in range(n_gaussians):

        stacked_p = geometry.Ellipsoid(position=mu[:, i], radius=stdev[:, i]).compute_xyz()

        points = np.vstack((points, stacked_p))

        stacked_rgb = geometry.Ellipsoid(position=mu[:, i], radius=stdev[:, i]).compute_rgb()

        points_rgb = np.vstack((points_rgb, stacked_rgb))
        v = vis_util.visualize_pointCloud_in_xyz(point_cloud=points, rgb=points_rgb)

mean = np.array([0,0,0])
cov = 0.05*np.eye(3)
data = points_ellipsoid(position=mean, radius=[0.5,0.3,0.9], num=100)

rgb = pptk.rand(10000, 3)
v = pptk.viewer(data, rgb)
v.set(point_size=0.005)

gmm = mixture.GaussianMixture(n_components=1, covariance_type='diag')
gmm.fit(data)
visualize_3d_gmm(data,rgb, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)



print("Covariances : ", gmm.covariances_)
print("Mean : ", gmm.means_)
print("Weights : ", gmm.weights_)
