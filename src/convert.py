import os
import numpy as np
# import cv2

color_map = {0: (0, 0, 0),  # black
             1: (0, 0, 255),  # blue -> first echo
             2: (0, 255, 0),  # green -> second echo
             3: (255, 0, 0),  # red -> first and second echo
             4: (255, 255, 0),  # yellow -> third echo
             5: (255, 0, 255),  # purple -> first and third echo
             6: (0, 255, 255),  # cyan -> second and third echo
             7: (255, 255, 255)}  # white

input_dir = 'data/class8/5/train'
output_dir = 'data/train'

file_list = os.listdir(input_dir)

# 原先的数据是 7 个通道，3组距离和脉宽，一个标签通道
# 现在去掉第三组距离和脉宽，只保留前两组，并且标签通道从8类(0-7)变为4类(0-3)

for file in file_list:
    data = np.load(os.path.join(input_dir, file)).reshape(7, 128, 1200)
    # # distance_1 intensity_1 distance_2 intensity_2 distance_3 intensity_3 label
    # # use cv2 to show the 7 channels
    # cv2.imshow("distance_1", data[0])
    # cv2.imshow("intensity_1", data[1])
    # cv2.imshow("distance_2", data[2])
    # cv2.imshow("intensity_2", data[3])
    # cv2.imshow("distance_3", data[4])
    # cv2.imshow("intensity_3", data[5])
    # cv2.imshow("label", data[6])
    
    # # use color_map to show the label
    # label_to_show = np.zeros((128, 1200, 3), dtype=np.uint8)
    # for i in range(128):
    #     for j in range(1200):
    #         label_to_show[i, j] = color_map[data[6, i, j]]
    # cv2.imshow("label", label_to_show)
    # cv2.waitKey(0)

    image = data[:4, :, :]
    label = data[6, :, :]

    label = np.array(label, dtype=np.int32).reshape(1, 128, 1200)
    label = label & 3

    image = image.reshape(4, 128, 1200)
    label = label.reshape(1, 128, 1200)

    data = np.concatenate((image, label), axis=0)

    print(data.shape)
    np.save(os.path.join(output_dir, file), data)
