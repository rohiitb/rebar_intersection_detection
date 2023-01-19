import os
import glob
import matplotlib.pyplot as plt
from src.vision import camera
import numpy as np



class ImgDataBase:
    '''
    This class loads the information of the images and their corresponding camera parameters
        1. useful methods:
            a. self.__getitem__(self.idx) # this will read the image based on index
            b. self.get(self,img_name) # this will read the image based on image name (xxx.JPG)
            c. self.get_name(self,idx) # get the image name based on index
            d. get_camera_attr(self,key,attr): # obtain the camera parameters
    '''
    def __init__(self,root='data',camera_params_file = '/Merged_2_3_4_calibrated_camera_parameters.txt',
    point_cloud_file = '/Merged_2_3_4_group1_densified_point_cloud.xyz'):
        self.root = root
        self.camera_params_file = camera_params_file
        self.point_cloud_file = point_cloud_file
        self.img_path = self.get_image_path()
        self.camera_dict = self.get_cameraParameters()
        # self.point_cloud = self.get_pointCloud()
    def get_image_path(self):
        img_path = glob.glob(self.root + "/undistorted_images/*.JPG")
        return img_path
    def get_cameraParameters(self):
        return camera.getCameraParametersDict(self.root + self.camera_params_file)
    def get_pointCloud(self):
        filename = self.root + self.point_cloud_file
        pointCloud_coord = []
        with open(filename,'r') as f:
            line = f.readline().split()
            while line:
                pointCloud_coord.append(line[:3])
                line = f.readline().split()
        return np.array(pointCloud_coord).astype('float')

    def __getitem__(self,idx):
        return plt.imread(self.img_path[idx])
    def show(self, idx: object) -> object:
        if isinstance(idx,int):
            im = plt.imread(self.img_path[idx])
            plt.imshow(im)
        elif isinstance(idx,str):
            img = self.root+"/undistorted_images/"+idx
            if img not in self.img_path:
                print("image not found")
            else:
                im = self.get(idx)
                plt.imshow(im)
        else:
            raise("should be either integer or string")
    def get(self,img_name):
        im = plt.imread(self.root+"/undistorted_images/"+img_name)
        return im
    def get_name(self,idx):
        return self.img_path[idx][-12:]
    
    def get_camera_attr(self,key,attr):
        '''
        obtainning the camera parameters:
            key:    either index or image name
            attr:   camera parameters. 
                    It can be set to 't', 'R', 'K'. 
                    Set to 'all' if want all of the three.
        '''
        if isinstance(key,str):
            imgName = key
            camera_params = self.camera_dict[key]
            return camera_params.t
        elif isinstance(key,int):
            imgName = self.get_name(key)
        camera_params = self.camera_dict[imgName]
        if attr == 't':
            return camera_params.t
        elif attr == 'R':
            return camera_params.R
        elif attr == 'K':
            return camera_params.K
        elif attr == 'all':
            return camera_params
        else:
            print('camera attributes not recognized')
            raise
    
    def __len__(self):
        return len(self.img_path)




        
