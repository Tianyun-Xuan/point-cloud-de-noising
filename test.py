import numpy as np
import h5py
import cv2
import os
import time

# image_table = np.load("/Users/xavier/Documents/calibration/point-cloud-de-noising/data/class8/train/361.npy").reshape(7, 128,1200)
# # 生成图像 7 * 128 * 1200
# # distance_1 intensity_1 distance_2 intensity_2 distance_3 intensity_3 label
# # use cv2 to show the 7 channels
# cv2.imshow("distance_1", image_table[0])
# cv2.imshow("intensity_1", image_table[1])
# cv2.imshow("distance_2", image_table[2])
# cv2.imshow("intensity_2", image_table[3])
# cv2.imshow("distance_3", image_table[4])
# cv2.imshow("intensity_3", image_table[5])
# cv2.imshow("label", image_table[6])
# cv2.waitKey(0)

# for i in range(7):
#     print(np.min(image_table[i]), np.max(image_table[i]))


# with h5py.File("/Users/xavier/Documents/calibration/point-cloud-de-noising/data/8/train/1439.hdf5", "r", driver='core') as hdf5:
#     # for channel in self.channels:
#     distance_1 = hdf5.get('distance_1')[()]
#     intensity_1 = hdf5.get('intensity_1')[()]
#     distance_2 = hdf5.get('distance_2')[()]
#     intensity_2 = hdf5.get('intensity_2')[()]
#     distance_3 = hdf5.get('distance_3')[()]
#     intensity_3 = hdf5.get('intensity_3')[()]
#     labels = hdf5.get('label')[()]

#     # print每个的最大最小值
#     print(np.min(distance_1), np.max(distance_1))
#     print(np.min(intensity_1), np.max(intensity_1))
#     print(np.min(distance_2), np.max(distance_2))
#     print(np.min(intensity_2), np.max(intensity_2))
#     print(np.min(distance_3), np.max(distance_3))
#     print(np.min(intensity_3), np.max(intensity_3))
#     print(np.min(labels), np.max(labels))


file_list = os.listdir("data/class8/train")

pluse_max = 0
for file in file_list:
    data = np.load("data/class8/train/" + file).reshape(7, 128, 1200)
    pluse_max = max(pluse_max, np.max(data[1]), np.max(data[3]), np.max(data[5]))

print(pluse_max)
