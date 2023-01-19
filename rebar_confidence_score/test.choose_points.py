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

# data directory
root = "data"

image_dataset = database.ImgDataBase(root = root)

idx = 10

im = image_dataset[idx]
print(image_dataset.get_name(idx))

plt.imshow(im)
plt.show()
