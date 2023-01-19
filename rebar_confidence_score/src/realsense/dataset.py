import rosbag
import matplotlib.pyplot as plt
import numpy as np
import glob

def bag2numpy(root='data/realsense/',filename='test.bag'):
    if root[-1] != '/':
        root = root + '/'
    bag = rosbag.Bag(root+filename)
    img_color = []
    count = 0
    loop_count =0
    idx = 0
    topics= None
    topics = ['/device_0/sensor_0/Depth_0/info/camera_info',
            '/device_0/sensor_1/Color_0/info/camera_info',
            '/device_0/sensor_1/Color_0/image/data',
            '/device_0/sensor_0/Depth_0/image/data',
            '/device_0/sensor_0/option/Depth_Units/value']
    for topic, msg, t in bag.read_messages(topics):
        # loop_count +=1
        if topic == topics[0]:
            H = msg.height
            W = msg.width
            K = msg.K
        if topic == topics[2]:
            img_color = np.frombuffer(msg.data, dtype=np.uint8).reshape(H,W,3)
            count +=1
        if topic == topics[3]:
            img_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(H,W)
        if topic == topics[4]:
            depth_unit = msg

        # if loop_count > 500: # delete later
        #     break
        # print(topic)


        if count % 15 ==1:
            with open(root +'color/'+str(idx)+'.npy', 'wb') as f:
                np.save(f, img_color)
                print('saved to',root +'color/'+str(idx)+'.npy')
            with open(root +'depth/'+str(idx)+'.npy', 'wb') as f:
                np.save(f, img_depth)
                print('saved to',root +'depth/'+str(idx)+'.npy')
            idx +=1
    bag.close()

    with open(root +'depth/K.npy', 'wb') as f:
        np.save(f, K)
        print('saved to',root +'depth/K.npy')

class Dataset:
    def __init__(self,root='data/realsense/'):
        self.root = root if root[-1]=='/' else root+'/'
        self.img_color_paths = glob.glob(self.root+'color/*.npy')
        self.img_depth_paths = glob.glob(self.root+'depth/*.npy')
        self.K = self.getK().reshape(3,3)
    def get(self,idx,attr):
        if attr == 'color':
            with open(self.img_color_paths[idx],'rb') as f:
                print(self.img_color_paths[idx])
                im = np.load(f)
        elif attr == 'depth':
            with open(self.img_depth_paths[idx],'rb') as f:
                im = np.load(f)
                print(self.img_depth_paths[idx])

        return im
    def getK(self):
        with open(self.root+'depth/K.npy','rb') as f:
            K = np.load(f)
        return K
    def __len__(self):
        return len(self.img_color_paths)
