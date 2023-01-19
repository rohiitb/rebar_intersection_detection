#!/usr/bin/env python3

import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import matplotlib as mpl
import src.realsense.dataset as rld
import src.realsense.camera as rlc
import src.mymodel.gmm as mygmm
import torch
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')



def get_pose(color,depth,K):

        # RGB filtering ###########################
        # data = pd.read_csv("data/colours_rgb_shades.csv", usecols = [1,2,3,4])
        # data.head()

        # numpied_data = data.to_numpy()[:,:3]
        # # print("Numpied Data : ", numpied_data)

        # numpied_vals = data.to_numpy()[:,-1:].reshape(-1)
        # # print("NUmpied Values : ", numpied_vals)

        # model = RandomForestClassifier()
        # model.fit(numpied_data, numpied_vals)
        ###################################################################

        start_time = time.time()
        H,W = color.shape[:2]
        # print(H,W)
        dilation = 100
         # selet 1 point for every dilation number of points
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')

        segmentation_start_time = time.time()
        # simple plane segmentation
        xyz_full,rgb_full = rlc.project_rgbd2pcd(color,depth,K,dilation=dilation) # project to point cloud

        idx = xyz_full[:,2]>0.001
        gmm_start_time = time.time()

        xyz = xyz_full[idx,:]; rgb = rgb_full[idx,:]
        # xyz = xyz[::dilation,:]; rgb = rgb[::dilation,:]   
        xy = xyz[:,:2]; z=xyz[:,2]
        reg = linear_model.LinearRegression()
        reg.fit(xy,z)

        class_binary = (z - (xy @ reg.coef_ + reg.intercept_) ) < -0.020
        xyz_top = xyz[class_binary,:]
        rgb_top = rgb[class_binary,:]

        # RGB filter
        # rgb_binary = model.predict(rgb_top) == 1
        # xyz_rgb_filtered = xyz_top[rgb_binary,:]
        # rgb_rgb_filtered = rgb_top[rgb_binary,:]
        # color_top, depth_top = rlc.project_pcd2rgb(xyz_rgb_filtered,rgb_rgb_filtered,H,W,K)
        ############################

        color_top, depth_top = rlc.project_pcd2rgb(xyz_top,rgb_top,H,W,K)

        # prepare for GMM
        y_idx = np.tile(np.expand_dims(np.arange(H),1),(1,W))
        x_idx = np.tile(np.arange(W),(H,1))
 
        binary_depth = depth_top > 0.001
        # gmm_x = x_idx[binary_depth].reshape(-1)
        # gmm_y = y_idx[binary_depth].reshape(-1)
        X = torch.from_numpy(np.c_[x_idx[binary_depth].reshape(-1),y_idx[binary_depth].reshape(-1)]).unsqueeze(0).to(torch.double).to(device)

        # GMM prediction
        
        # X = torch.from_numpy(X).unsqueeze(0).to(torch.double).to(device)
        my_gmm = mygmm.GMM_torch(n_components=30, max_iter=30)
        gmm_fit_start_time = time.time()
        my_gmm.fit(X)
        gmm_filter_start_time = time.time()
        my_gmm.filter_GMM(X,filter_ratio=4)
        gmm_merge_start_time = time.time()
        my_gmm.merge_GMM(X)
        
        pose_start_time = time.time()
        # lines in GMM
        lines = []
        for i, (class_idx, mean, covar,angle,u,v) in enumerate(my_gmm.merge_list):
            p2 = mean - torch.tensor([u[1],-u[0]])*v[1]
            p3 = mean + torch.tensor([u[1],-u[0]])*v[1]  
            lines.append((p2,p3))
        
        # compute the pose (intersections and vectors1 and vectors2)
        intersections = []
        vectors1 = []
        vectors2 = []
        vectors3d_1 = []
        vectors3d_2 = []
        th_vec1_vec2 = []
        for i in range(len(lines)):
            for j in range(i+1,len(lines)):
                (x1,y1),(x2,y2) = lines[i]
                (x3,y3),(x4,y4) = lines[j]
                px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
                py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))   
                thres = 30
                m1 = torch.arctan((y1-y2)/(x1-x2))
                m2 = torch.arctan((y3-y4)/(x3-x4))
                diff_m = min(abs(m1 - m2),np.pi-abs(m1-m2))
                
                if px < max(min(x1,x2),min(x3,x4))-thres or px > min(max(x1,x2),max(x3,x4))+thres:
                    continue
                elif  diff_m < np.pi/2*0.75: # filter out the non-perpendicular pair
                    continue
                # elif px < color.shape[1] and py < color.shape[0] and int(model.predict(np.array(color[int(py), int(px), :]).reshape(1,3))) == 0:      # RGB filtering
                #     continue
                else:
                    # 2d space
                    intersections.append(np.array([px.numpy(),py.numpy()]))
                    v1 = torch.tensor([x2-x1,y2-y1]); v1 = v1/torch.linalg.norm(v1)
                    v2 = torch.tensor([x4-x3,y4-y3]); v2 = v2/torch.linalg.norm(v2)
                    vectors1.append(v1.numpy()); vectors2.append(v2.numpy());#th_vec1_vec2.append(np.arccos(v1@v2))

#                     # 3d space
#                     u1_temp = np.linspace(x1,x2,num=100).astype(np.int)
#                     v1_temp = np.linspace(y1,y2,num=100).astype(np.int)
#                     u1 = u1_temp[(u1_temp<W)&(v1_temp<H)]; v1 = v1_temp[(u1_temp<W)&(v1_temp<H)]
#                     d1 = depth_top[v1,u1]
#                     xyz1 = rlc.project_pixel2pcd(u1,v1,d1,K)
#                     xyz1 = xyz1[(xyz1[:,0]>0.001)|(xyz1[:,0]<-0.001),:]
#                     x1 = xyz1[:,0];y1 = xyz1[:,1];z1 = xyz1[:,2];t1 = np.arange(len(x1)).reshape(-1,1)
#                     if len(x1)>20:
#                         if x1.max() - x1.min() > 0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t1, x1);kx1 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t1, x1);kx1 = reg.coef_
#                         if y1.max() - y1.min() > 0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t1, y1);ky1 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t1, y1);ky1 = reg.coef_
#                         if z1.max() - z1.min() > 0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t1, z1);kz1 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t1, z1);kz1 = reg.coef_
#                     elif len(x1)>1:
#                         reg = LinearRegression().fit(t1, x1);kx1 = reg.coef_
#                         reg = LinearRegression().fit(t1, y1);ky1 = reg.coef_
#                         reg = LinearRegression().fit(t1, z1);kz1 = reg.coef_
#                     else:
#                         kx1=ky1=kz1=0
#                     vector1 = np.array([kx1,ky1,kz1]); 
#                     if kx1+ky1+kz1 == 0:
#                         vector1 = np.array([0,0,0])
#                     else:
#                         vector1 = vector1/np.linalg.norm(vector1);
#                     vectors3d_1.append(vector1)

#                     u2_temp = np.linspace(x3,x4,num=100).astype(np.int)
#                     v2_temp = np.linspace(y3,y4,num=100).astype(np.int)
#                     u2 = u2_temp[(u2_temp<W)&(v2_temp<H)]; v2 = v2_temp[(u2_temp<W)&(v2_temp<H)]
#                     d2 = depth_top[v2,u2]
#                     xyz2 = rlc.project_pixel2pcd(u2,v2,d2,K)
#                     xyz2 = xyz2[(xyz2[:,0]>0.001)|(xyz2[:,0]<-0.001),:]
#                     x2 = xyz2[:,0];y2 = xyz2[:,1];z2 = xyz2[:,2];t2 = np.arange(len(x2)).reshape(-1,1)
#                     if len(x2)>20:
#                         if x2.max() - x2.min() > 0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t2, x2);kx2 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t2, x2);kx2 = reg.coef_
#                         if y2.max() - y2.min() >0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t2, y2);ky2 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t2, y2);ky2 = reg.coef_
#                         if z2.max() - z2.min() > 0.02:
#                             ransac = linear_model.RANSACRegressor();ransac.fit(t2, z2);kz2 = ransac.estimator_.coef_
#                         else:
#                             reg = LinearRegression().fit(t2, z2);kz2 = reg.coef_                
#                     elif len(x2)>1:
#                         reg = LinearRegression().fit(t2, x2);kx2 = reg.coef_
#                         reg = LinearRegression().fit(t2, y2);ky2 = reg.coef_
#                         reg = LinearRegression().fit(t2, z2);kz2 = reg.coef_
#                     else:
#                         kx2=ky2=kz2=0
#                     vector2 = np.array([kx2,ky2,kz2]); 
#                     if kx2+ky2+kz2 == 0:
#                         vector2 = np.array([0,0,0])
#                     else:
#                         vector2 = vector2/np.linalg.norm(vector2);
#                     vectors3d_2.append(vector2)

        # intersections = torch.tensor(intersections)
        # vectors1 = torch.tensor(vectors1)
        # vectors2 = torch.tensor(vectors2)
        # th_vec1_vec2 = torch.tensor(th_vec1_vec2)
        # th_vec1_vec2[th_vec1_vec2>np.pi/2] = np.pi - th_vec1_vec2[th_vec1_vec2>np.pi/2]

        print(f"TOTAL DETECTION TIME = {-start_time+time.time():.3f} s")
        print(f"\t-data loading time = {-start_time+segmentation_start_time:.3f} s")
        print(f"\t-plane segment time = {-segmentation_start_time + gmm_start_time:.3f} s")
        print(f"\t-gmm time  = {-gmm_start_time+pose_start_time:.3f} s")
        print(f"\t\ta) gmm data prepare time  = {-gmm_start_time+gmm_fit_start_time:.3f} s")
        print(f"\t\tb) gmm iteration time  = {-gmm_fit_start_time+gmm_filter_start_time:.3f} s")
        print(f"\t\tc) gmm filtering time  = {-gmm_filter_start_time+gmm_merge_start_time:.3f} s")
        print(f"\t\td) gmm merging time  = {-gmm_start_time+pose_start_time:.3f} s")
        print(f"\t-pose time = {-pose_start_time+time.time():.3f} s")


        return intersections, vectors1, vectors2#, color_top
