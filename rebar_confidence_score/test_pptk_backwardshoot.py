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



point_proj = 0
uv_projected, if_inside = camera.projection_world2img(point_proj,idx,image_dataset)

