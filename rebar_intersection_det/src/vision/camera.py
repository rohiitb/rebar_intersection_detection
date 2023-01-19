from src.vision.util import str2double
import numpy as np
from src.visual3d import geometry
from sklearn.neighbors import NearestNeighbors
from src.vision import database, camera

class CameraParameters:
    def __init__(self, imgName,W,H,K,distortion_radial,distortion_tangential,t,R):
        self.imgName = imgName
        self.W = W
        self.H = H
        self.distortion_radial = str2double(distortion_radial)
        self.distortion_tangential = str2double(distortion_tangential)
        self.K = str2double(K)
        self.t = str2double(t)
        self.R = str2double(R)

    def __str__(self):
        s = f"imgName = {self.imgName}\n"
        s = s + f"W = {self.W}\n"
        s = s + f"H = {self.H}\n"
        s = s + f"distortion_radial = {self.distortion_radial}\n"
        s = s + f"distortion_tangential = {self.distortion_tangential}\n"
        s = s + f"K = {self.K}\n"
        s = s + f"t = {self.t}\n"
        s = s + f"R = {self.R}\n"
        
        return s

def getCameraParametersDict(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()
    idx = 0
    camera_parameters_dict = {}
    while idx < len(lines):
        line = lines[idx]
        if "JPG" in line:
            imgName = line[:12]
            H = int(line[12:17])
            W = int(line[17:])
            K = [];distortion_radial = [];distorsion_tangential=[];t=[];R=[]
            K.append(lines[idx+1][:-1].split(' '))
            K.append(lines[idx+2][:-1].split(' '))
            K.append(lines[idx+3][:-1].split(' '))
            distortion_radial = lines[idx+4][:-1].split(' ')
            distortion_tangential = lines[idx+5][:-1].split(' ')
            t = lines[idx+6][:-1].split(' ')
            R.append(lines[idx+7][:-1].split(' '))
            R.append(lines[idx+8][:-1].split(' '))
            R.append(lines[idx+9][:-1].split(' '))
            camera_params = CameraParameters(imgName,W,H,K,distortion_radial,distortion_tangential,t,R)
            camera_parameters_dict[imgName] = camera_params
            idx += 9
        idx+=1
    return camera_parameters_dict

def projection_img2world_line(uv,camera_params,z=-2.0):
    t = np.array(camera_params.t)
    t = np.expand_dims(t,axis = 1)
    R = np.array(camera_params.R)
    K = np.array(camera_params.K)
    M = K @ np.block([R,-R@t])
    u = uv[0];v = uv[1]
    A = np.array([
        [M[0,0] - u*M[2,0], M[0,1] - u*M[2,1]],
        [M[1,0] - v*M[2,0], M[1,1] - v*M[2,1]]
        ])
    b1 = np.array([
        [M[0,2] - u*M[2,2]],
        [M[1,2] - v*M[2,2]]
        ])
    b2 = np.array([
        [M[0,3] - u*M[2,3]],
        [M[1,3] - v*M[2,3]]
        ])
    xy = np.linalg.solve(A,-b1*z-b2)

    start = t
    end = np.append(xy,z)

    return start, end

def projection_img2world_point(uv,idx,image_dataset,point_cloud_xyz):
    # idx = 0; uv=[ 2723,2067]

    # compute projection line
    camera_params = image_dataset.get_camera_attr(idx,'all')
    start, end = camera.projection_img2world_line(uv,camera_params,z=np.min(point_cloud_xyz[:,2]))
    projection_line = geometry.Line(start=start,end=end)

   
    # compute the projection points
    n_neighbors = 1000
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(point_cloud_xyz)
    distances, indices = nbrs.kneighbors(projection_line.xyz)

    min_indices = np.argmin(distances,axis=0)
    indices_filterd = indices[min_indices,range(n_neighbors)]
    points_filtered = geometry.PointCloud(point_cloud_xyz[indices_filterd,:])

    n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points_filtered.xyz)
    distances, indice = nbrs.kneighbors([camera_params.t])
    ind_closet = np.argmin(distances)
    projection_point = geometry.Sphere(position=points_filtered.xyz[ind_closet,:],radius=0.003,color='red',num=20)

    return projection_point, projection_line, points_filtered

def projection_world2img(position_3d,idx,image_dataset):
    camera_params = image_dataset.get_camera_attr(idx,'all')
    H = camera_params.H; W = camera_params.W
    point_proj_HT = np.append(position_3d,1.0)
    R = np.array(camera_params.R)
    K = np.array(camera_params.K)
    t = np.array(camera_params.t)
    t = np.expand_dims(t,axis=1)
    M = np.block([R,-R@t])
    x = K @ M @ point_proj_HT; x = x/x[2]; x = x.astype('int')
    if_inside_img = False
    if x[0]>=0 and x[0]<H and x[1]>=0 and x[1]<W:
        if_inside_img = True

    return x[:2], if_inside_img