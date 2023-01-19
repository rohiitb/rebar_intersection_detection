from matplotlib.colors import rgb2hex
import numpy as np
import torch

def project_rgbd2pcd(img_rgb,img_d,K,dilation=1):
    H,W,_ = img_rgb.shape
    dilation_idx = np.random.rand(H*W)<1.0/dilation
    u = np.repeat(np.arange(W),H)[dilation_idx]
    # v = (np.ones((W,1))*np.arange(H)).reshape(-1).astype(np.int)
    v = np.tile(np.expand_dims(np.arange(H),1),(W,1)).reshape(-1)[dilation_idx]
    one = np.ones(u.shape).astype(np.int)
    uv1 = np.r_['0,2',u,v,one]
    # print(uv1.shape)
    K_inv = np.linalg.inv(K)
    d = img_d[v,u]
    xyz = K_inv@uv1

    lamb = d/xyz[2,:]
    xyz[0,:] = lamb * xyz[0,:]
    xyz[1,:] = lamb * xyz[1,:]
    xyz[2,:] = d

    rgb= img_rgb[v,u,:]

    return xyz.T, rgb

def project_pixel2pcd(u,v,d,K):
    one = np.ones(u.shape).astype(np.int)
    uv1 = np.r_['0,2',u,v,one]
    K_inv = np.linalg.inv(K)
    xyz = K_inv@uv1
    lamb = d/xyz[2,:]
    xyz[0,:] = lamb * xyz[0,:]
    xyz[1,:] = lamb * xyz[1,:]
    xyz[2,:] = d
    return xyz.T

def project_pcd2rgb(xyz,rgb,H,W,K):
    color = np.ones((H,W,3)).astype(np.uint8)*255
    depth = np.zeros((H,W))
    uv1_l = K @ xyz.T
    uv1 = (uv1_l/uv1_l[2,:]).astype(np.uint16)
    color[uv1[1,:],uv1[0,:],:] = rgb
    depth[uv1[1,:],uv1[0,:]] = xyz[:,2]

    return color, depth




