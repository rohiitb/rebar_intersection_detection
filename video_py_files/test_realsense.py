
import rosbag
import matplotlib.pyplot as plt
import numpy as np

bag = rosbag.Bag('data/realsense/test.bag')
count = 0
# for topic, msg, t in bag.read_messages(['/device_0/sensor_1/Color_0/image/data',
#                                         '/device_0/sensor_0/Depth_0/image/data']):
topics = []
count_loop = 0
# topics = ['/device_0/sensor_0/Depth_0/info/camera_info','/device_0/sensor_1/Color_0/info/camera_info','/device_0/sensor_1/Color_0/image/data','/device_0/sensor_0/Depth_0/image/data']
for topic, msg, t in bag.read_messages():
    if topic == topics[0]:
        H = msg.height
        W = msg.width
        K = msg.K
    if topic == topics[2]:
        img_color = np.frombuffer(msg.data, dtype=np.uint8).reshape(H,W,3)
        # count +=1
    if topic == topics[3]:
        img_depth = np.frombuffer(msg.data, dtype=np.uint8).reshape(H,W,2)

    if count_loop  > 500:
        break
    count_loop +=1

    if count >1:
        break
bag.close()

plt.subplot(121)
plt.imshow(img_depth[:,:,0])
print((img_depth[:,:,0]))
plt.subplot(122)
plt.imshow(img_depth[:,:,1])
print(np.max(img_depth[:,:,1]))

plt.show()
