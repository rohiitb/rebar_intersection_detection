from logging import raiseExceptions
import numpy as np
import pptk

class Geometry:
    def __init__(self):
        self.xyz = []
        self.rgb = []
        self.color = 'green'
    def add_oneGeom(self,geom):
        if len(self.xyz)>0:
            if len(geom.xyz) != 0:
                self.xyz = np.vstack((self.xyz,geom.xyz))
                self.rgb = np.vstack((self.rgb,geom.rgb))
        else:
            self.xyz = geom.xyz
            self.rgb = geom.rgb
    def show(self,lookat = [0,0,-2]):
        v = pptk.viewer(self.xyz)
        v.attributes(self.rgb / 255.)
        v.set(lookat=lookat)
        return v

    def add(self,*geoms):
        for geom in geoms:
            self.add_oneGeom(geom)
    def compute_rgb(self):
            if self.color == 'green':
                r0 = 0; g0 = 255; b0 = 0
            elif self.color == 'red':
                r0 = 255; g0 = 0; b0 = 0
            elif self.color == 'blue':
                r0 = 0; g0 = 0; b0 = 255
            else:
                if len(self.color) < 3:
                    print('color not identified')
                    raise 
                elif len(self.color)==3:
                    r0, g0, b0 = self.color
                else:
                    self.rgb = self.color
                    return
            num_points = len(self.xyz)
            self.rgb = np.c_[np.ones(num_points)*r0,np.ones(num_points)*g0,np.ones(num_points)*b0]

class Line(Geometry):
    def __init__(self,start = np.array([0,0,0]), end = np.array([0,0,1]),num=400, linewidth = None,color = "green"):
        super().__init__()
        self.start = np.array(start)
        self.end = np.array(end)
        self.num = num
        self.linewidth = linewidth
        self.color = color
        self.compute_xyz()
        self.compute_rgb()

    def compute_xyz(self):
        xc = np.linspace(self.start[0],self.end[0],num = self.num)
        yc = np.linspace(self.start[1],self.end[1],num = self.num)
        zc = np.linspace(self.start[2],self.end[2],num = self.num)
        
        if self.linewidth == None:
            self.xyz = np.c_[xc,yc,zc]
        else:
            num_circle = 30 # number of points to form a circle
            th = np.linspace(0,np.pi*2,num_circle) 
            x_circle = self.linewidth * np.cos(th)
            y_circle = self.linewidth * np.sin(th)
            z_circle = np.zeros(num_circle)
            v_norm = (self.end - self.start)
            v_norm = v_norm /np.linalg.norm(v_norm)
            v_0 = np.array([0,0,1])
            rotation_ang = np.arccos(v_norm @ v_0)
            rotation_axis = np.cross(v_norm,v_0)
            rotm = self.axis_angle_rotation(rotation_axis,rotation_ang)

            x = np.array([]);y=x;z=x
            for i in range(self.num):
                pass
            raise NotImplementedError

    def axis_angle_rotation(self,w,th):
        K = np.array([[0, -w[2], w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

class Sphere(Geometry):
    def __init__(self,position = [0,0,0],radius = 0.01,num = 5,color='green'):
        super().__init__()
        self.radius = radius
        self.position = position
        self.num = num
        self.color = color
        self.compute_xyz()
        self.compute_rgb()
    def compute_xyz(self):
        phi = np.linspace(0,np.pi*2,self.num)
        phi = np.expand_dims(phi,axis=0)
        th = np.linspace(0,np.pi*2,self.num)
        th = np.expand_dims(th,axis=0)
        x = (self.radius * (np.cos(phi.T) @ np.sin(th)) + self.position[0]).reshape(-1)
        y = (self.radius * (np.sin(phi.T) @ np.sin(th)) + self.position[1]).reshape(-1)
        z = (self.radius * (np.ones((self.num,1)) @ np.cos(th))+ self.position[2]).reshape(-1)
        self.xyz = np.c_[x,y,z]
  

class Rectangle(Geometry):
    def __init__(self,H,W,center_positoin,R):
        super().__init__()
    


class PointCloud(Geometry):
    def __init__(self,xyz=[],color = 'green'):
        super().__init__()
        self.xyz = np.array(xyz)
        self.color = np.array(color)
        self.compute_rgb()
    

