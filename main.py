import h5py
import numpy as np

import cv2
import os
import time

color_map = {0: (0, 0, 0),  # black
             1: (0, 0, 255),  # blue -> first echo
             2: (0, 255, 0),  # green -> second echo
             3: (255, 0, 0),  # red -> first and second echo
             4: (255, 255, 0),  # yellow -> third echo
             5: (255, 0, 255),  # purple -> first and third echo
             6: (0, 255, 255),  # cyan -> second and third echo
             7: (255, 255, 255)}  # white


def load_hdf5_file(filename, save_name):
    points = None
    with h5py.File(filename, "r", driver='core') as hdf5:
        # for channel in self.channels:
        sensorX_1 = hdf5.get('sensorX_1')[()]
        sensorY_1 = hdf5.get('sensorY_1')[()]
        sensorZ_1 = hdf5.get('sensorZ_1')[()]
        distance_m_1 = hdf5.get('distance_m_1')[()]
        intensity_1 = hdf5.get('intensity_1')[()]
        labels_1 = hdf5.get('labels_1')[()]

        print(sensorX_1.shape)

        points = np.stack((sensorX_1, sensorY_1, sensorZ_1,
                          distance_m_1, intensity_1, labels_1), axis=-1)
    print(points.shape)
    points = points.reshape(-1, 6)
    np.savetxt(save_name, points, delimiter=",")


def generate_hdf5_file(filename):
    points = np.loadtxt("point.txt", delimiter=",")
    with h5py.File(filename, "w", driver='core') as hdf5:
        hdf5.create_dataset('sensorX_1', data=points[:, 0])
        hdf5.create_dataset('sensorY_1', data=points[:, 1])
        hdf5.create_dataset('sensorZ_1', data=points[:, 2])
        hdf5.create_dataset('distance_m_1', data=points[:, 3])
        hdf5.create_dataset('intensity_1', data=points[:, 4])
        hdf5.create_dataset('labels_1', data=points[:, 5])


# def generate_marker_points():

#     def get_useful_data():
#         background = np.loadtxt("rayz/txt/total/back.txt").reshape(-1, 9)
#         target = np.loadtxt("rayz/txt/total/target.txt").reshape(-1, 9)

#         # x y z row col pluse echo theta phi
#         # 0 1 2  3   4   5   6    7    8
#         background[:, -1] = 0
#         target[:, -1] = 1
#         background[:, -2] = np.linalg.norm(background[:, :3], axis=1)
#         target[:, -2] = np.linalg.norm(target[:, :3], axis=1)

#         # useful
#         # x y z distance intensity label
#         useful_background = background[:, [0, 1, 2, -2, -4, -1]]
#         useful_target = target[:, [0, 1, 2, -2, -4, -1]]

#         useful_data = np.concatenate((useful_background, useful_target), axis=0)

#         return useful_data.astype(np.float32)

#     useful_data = get_useful_data()
#     useful_data = useful_data.reshape(-1, 6)

#     # 打乱点的顺序
#     np.random.shuffle(useful_data)
#     # 每次取出 32 x 400 个点 的 6个参数
#     total_num = useful_data.shape[0] // (32*400)
#     for i in range(total_num):
#         points = useful_data[i*32*400:(i+1)*32*400]
#         with h5py.File("rayz/train/point_{}.hdf5".format(i), "w", driver='core') as hdf5:
#             hdf5.create_dataset('sensorX_1', data=points[:, 0].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('sensorY_1', data=points[:, 1].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('sensorZ_1', data=points[:, 2].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('distance_m_1', data=points[:, 3].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('intensity_1', data=points[:, 4].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('labels_1', data=points[:, 5].reshape(
#                 32, 400), dtype=np.float32)


# def frame_to_show():
#     back = np.loadtxt("rayz/frame/back.txt").reshape(-1, 10)
#     target = np.loadtxt("rayz/frame/target.txt").reshape(-1, 10)
#     # x y z row col pluse echo theta phi frame_id
#     # 0 1 2  3   4   5   6    7    8     9

#     # x y z distance intensity label frame_id
#     back[:, -2] = 0
#     target[:, -2] = 1
#     back[:, -3] = np.linalg.norm(back[:, :3], axis=1)
#     target[:, -3] = np.linalg.norm(target[:, :3], axis=1)

#     useful_data = np.concatenate((back, target), axis=0)
#     useful_data = useful_data[:, [0, 1, 2, -3, -4, -2, -1]]

#     # 打乱点的顺序
#     np.random.shuffle(useful_data)
#     # 每次取出 32 x 400 个点 的 6个参数
#     total_num = useful_data.shape[0] // (32*400)
#     for i in range(total_num):
#         points = useful_data[i*32*400:(i+1)*32*400]
#         with h5py.File("rayz/test/point_{}.hdf5".format(i), "w", driver='core') as hdf5:
#             hdf5.create_dataset('sensorX_1', data=points[:, 0].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('sensorY_1', data=points[:, 1].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('sensorZ_1', data=points[:, 2].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('distance_m_1', data=points[:, 3].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('intensity_1', data=points[:, 4].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('labels_1', data=points[:, 5].reshape(
#                 32, 400), dtype=np.float32)
#             hdf5.create_dataset('frame_id', data=points[:, 6].reshape(
#                 32, 400), dtype=np.float32)

# generate_marker_points()
# load_hdf5_file("data/train/LidarImage_000000005.hdf5", "point_5.txt")
# load_hdf5_file("data/train/LidarImage_000000003.hdf5", "point_3.txt")
# load_hdf5_file("data/train/LidarImage_000000006.hdf5", "point_6.txt")
# frame_to_show()


# For CNN model
pixel_row = 128
pixel_col = 1024


def remove_stra_light(frame, distance):
    # X Y Z row col pluse echo theta phi Original_cloud_index

    # 去除小于distance的点云，并按顺序修改其后续回波的编号
    distance_mask = np.linalg.norm(frame[:, :3], axis=1) > distance
    result = frame[distance_mask]
    removed_points = frame[~distance_mask]
    print("Total points: {}, Remaining points: {}, Removed points: {}".format(
        frame.shape[0], result.shape[0], removed_points.shape[0]))
    removed_row_col_pair = removed_points[:, 3:5]
    for i in range(removed_row_col_pair.shape[0]):
        row, col = removed_row_col_pair[i]
        points = result[(result[:, 3] == row) & (result[:, 4] == col)]
        # 按照距离给回波编号
        tdistance = np.linalg.norm(points[:, :3], axis=1)
        # 排序
        index = np.argsort(tdistance)
        # 重新编号
        techo = np.arange(points.shape[0])
        points[:, 6] = techo[index]

    return result


def from_pcd_to_image():
    # X Y Z row col pluse echo theta phi Original_cloud_index
    mist_array = np.loadtxt("rayz/300_t.txt").reshape(-1, 10)
    back_array = np.loadtxt("rayz/300_b.txt").reshape(-1, 10)

    origin_id = 300
    frame_id = np.unique(mist_array[:, -1])
    theta_factor = pixel_row / (np.pi / 2)
    phi_factor = pixel_col / np.pi

    for id in frame_id:
        mist = mist_array[mist_array[:, -1] == id]
        back = back_array[back_array[:, -1] == id]

        mist[:, -1] = 101
        back[:, -1] = 100

        total = np.concatenate((mist, back), axis=0)
        # total = remove_stra_light(total, 2)
        distance = np.linalg.norm(total[:, :3], axis=1)
        # echo == 0 是第一次回波
        # echo == 1 是第二次回波
        # 基本少有第三次回波所以省略
        intensity_1 = total[:, 4] * (total[:, 6] == 0)
        intensity_2 = total[:, 4] * (total[:, 6] == 1)
        row = np.int32((total[:, 7] + np.pi / 4) * theta_factor)
        col = np.int32((total[:, 8] + np.pi / 2) * phi_factor)

        table = np.zeros((4, pixel_row, pixel_col))
        # 生成图像 4 * 128 * 512
        # distance intensity_1 intensity_2 label
        # 0: no label, 100: valid/clear, 101: mist
        # theta [-pi/4, pi/4] -> 128
        # phi [-pi/2, pi/2] -> 1024
        for i in range(total.shape[0]):
            # once marked as mist, never change
            if table[3, row[i], col[i]] != 101:
                table[3, row[i], col[i]] = total[i, -1]
                table[0, row[i], col[i]] = distance[i]
                table[1, row[i], col[i]] = intensity_1[i]
                table[2, row[i], col[i]] = intensity_2[i]

        # use cv2 to show the 4 channels
        cv2.imshow("distance", table[0])
        cv2.imshow("intensity_1", table[1])
        cv2.imshow("intensity_2", table[2])
        # use color_map to show the label
        label_to_show = np.zeros((pixel_row, pixel_col, 3), dtype=np.uint8)
        for i in range(pixel_row):
            for j in range(pixel_col):
                label_to_show[i, j] = color_map[table[3, i, j]]
        cv2.imshow("label", label_to_show)
        cv2.waitKey(0)

        with h5py.File("rayz/train/point_{}.hdf5".format(origin_id + id), "w", driver='core') as hdf5:
            hdf5.create_dataset('distance', data=table[0].reshape(
                pixel_row, pixel_col), dtype=np.float32)
            hdf5.create_dataset('intensity_1', data=table[1].reshape(
                pixel_row, pixel_col), dtype=np.float32)
            hdf5.create_dataset('intensity_2', data=table[2].reshape(
                pixel_row, pixel_col), dtype=np.float32)
            hdf5.create_dataset('label', data=table[3].reshape(
                pixel_row, pixel_col), dtype=np.float32)

        # debug save x y z label echo
        debug_array = total[:, [0, 1, 2, 3, 6]]
        debug_array[:, 3] = table[3, row, col]
        # if debug_array[:, 4] != 0 than let debug_array[:, 3] = 100
        debug_array[:, 3] = debug_array[:, 3] * \
            (debug_array[:, 4] == 0) + 100 * (debug_array[:, 4] != 0)
        np.savetxt("rayz/label/point_{}.txt".format(origin_id + id),
                   debug_array, delimiter=",")


# test tesult
def show_test_result_v1():
    predictions = np.load("test_predictions.npy")
    labels = np.load("test_labels.npy")

    # X Y Z row col pluse echo theta phi Original_cloud_index
    mist_array = np.loadtxt("rayz/300_t.txt").reshape(-1, 10)
    back_array = np.loadtxt("rayz/300_b.txt").reshape(-1, 10)

    origin_id = 300
    frame_id = np.unique(mist_array[:, -1])
    theta_factor = pixel_row / (np.pi / 2)
    phi_factor = pixel_col / np.pi

    frame_id = np.unique(mist_array[:, -1])
    for id in frame_id:
        mist = mist_array[mist_array[:, -1] == id]
        back = back_array[back_array[:, -1] == id]
        total = np.concatenate((mist, back), axis=0)
        row = np.int32((total[:, 7] + np.pi / 4) * theta_factor)
        col = np.int32((total[:, 8] + np.pi / 2) * phi_factor)

        pre_table = predictions[np.int32(id), :, :]
        label_table = labels[np.int32(id), :, :]

        total[:, -1] = pre_table[row, col]
        distance = np.linalg.norm(total[:, :3], axis=1)

        total[:, -1] = total[:, -1] * (distance < 15) * (total[:, 6] == 0)

        np.savetxt("rayz/res/point_pre_{}.txt".format((int(origin_id) + int(id))),
                   total)

        # total[:, -1] = label_table[row, col]

        # total[:, -1] = total[:, -1] * (distance < 15)* (total[:, 6] == 0)

        # np.savetxt("rayz/res/point_label_{}.txt".format((int(origin_id) + int(id))),
        #            total)


# from_pcd_to_image()
# show_test_result_v1()


# for 128*1200 strictly fixed lidar scan pattern

def generate_scan_pattern_image(origin_id):
    # X Y Z row col pluse echo theta phi Original_cloud_index
    back_array = np.loadtxt("rayz/8/src/{}_b.txt".format(origin_id)).reshape(-1, 10)
    mist_array = np.loadtxt("rayz/8/src/{}_t.txt".format(origin_id))

    if back_array.shape[0] == 0:
        print("No back array")
        return
    if mist_array.shape[0] != 0:
        mist_array = mist_array.reshape(-1, 10)
    else:
        mist_array = np.array([]).reshape(0, 10)

    frame_id = np.unique(back_array[:, -1])

    for id in frame_id:
        count_id = np.int32(np.int32(id) + np.int32(origin_id))
        mist = mist_array[mist_array[:, -1] == id]
        back = back_array[back_array[:, -1] == id]

        mist[:, -1] = 101
        back[:, -1] = 100

        total = np.concatenate((mist, back), axis=0)
        # total echo -1
        total[:, 6] = total[:, 6] - 1
        total = remove_stra_light(total, 2)
        np.savetxt("rayz/without_stra_light/{}.txt".format(count_id), total)

        image_table = np.zeros((7, 128, 1200))
        # 用3个bit来编码三次回波是否是水雾的情况
        for i in range(128):
            for j in range(1200):
                points = total[(total[:, 3] == i) & (total[:, 4] == j)]
                if points.shape[0] != 0:
                    first_echo = points[points[:, 6] == 0]
                    second_echo = points[points[:, 6] == 1]
                    third_echo = points[points[:, 6] == 2]
                    label = 0
                    if first_echo.shape[0] != 0:
                        image_table[0, i, j] = np.linalg.norm(first_echo[0, :3])
                        image_table[1, i, j] = first_echo[0, 4]
                        if first_echo[0, -1] == 101:
                            label = label | 1
                    if second_echo.shape[0] != 0:
                        image_table[2, i, j] = np.linalg.norm(second_echo[0, :3])
                        image_table[3, i, j] = second_echo[0, 4]
                        if second_echo[0, -1] == 101:
                            label = label | 2
                    if third_echo.shape[0] != 0:
                        image_table[4, i, j] = np.linalg.norm(third_echo[0, :3])
                        image_table[5, i, j] = third_echo[0, 4]
                        if third_echo[0, -1] == 101:
                            label = label | 4
                    image_table[6, i, j] = label

        # 生成图像 7 * 128 * 1200
        # distance_1 intensity_1 distance_2 intensity_2 distance_3 intensity_3 label
        # use cv2 to show the 7 channels
        # cv2.imshow("distance_1", image_table[0])
        # cv2.imshow("intensity_1", image_table[1])
        # cv2.imshow("distance_2", image_table[2])
        # cv2.imshow("intensity_2", image_table[3])
        # cv2.imshow("distance_3", image_table[4])
        # cv2.imshow("intensity_3", image_table[5])
        # cv2.imshow("label", image_table[6])

        # # use color_map to show the label
        # label_to_show = np.zeros((pixel_row, pixel_col, 3), dtype=np.uint8)
        # for i in range(pixel_row):
        #     for j in range(pixel_col):
        #         label_to_show[i, j] = color_map[image_table[6, i, j]]
        # cv2.imshow("label", label_to_show)
        # cv2.waitKey(0)

        with h5py.File("rayz/train/{}.hdf5".format(count_id), "w", driver='core') as hdf5:
            hdf5.create_dataset('distance_1', data=image_table[0].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('intensity_1', data=image_table[1].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('distance_2', data=image_table[2].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('intensity_2', data=image_table[3].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('distance_3', data=image_table[4].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('intensity_3', data=image_table[5].reshape(
                128, 1200), dtype=np.float32)
            hdf5.create_dataset('label', data=image_table[6].reshape(
                128, 1200), dtype=np.float32)

        # debug save x y z row col label echo
        debug_array = total[:, [0, 1, 2, 3, 4, 9, 6]]
        for point in debug_array:
            label = image_table[6, int(point[3]), int(point[4])]
            ref = 2**int(point[-1])
            point[-2] = int(label) & int(ref)
        np.savetxt("rayz/label/{}.txt".format(count_id), debug_array, delimiter=",")


def verify_hdf5(filename):
    points = None
    with h5py.File(filename, "r", driver='core') as hdf5:
        # for channel in self.channels:
        distance_1 = hdf5.get('distance_1')[()]
        intensity_1 = hdf5.get('intensity_1')[()]
        distance_2 = hdf5.get('distance_2')[()]
        intensity_2 = hdf5.get('intensity_2')[()]
        distance_3 = hdf5.get('distance_3')[()]
        intensity_3 = hdf5.get('intensity_3')[()]
        labels = hdf5.get('label')[()]

        print(distance_1.shape)

        # 合并成 7 * 128 * 1200
        image_table = np.stack((distance_1, intensity_1, distance_2,
                               intensity_2, distance_3, intensity_3, labels), axis=0)

        # 生成图像 7 * 128 * 1200
        # distance_1 intensity_1 distance_2 intensity_2 distance_3 intensity_3 label
        # use cv2 to show the 7 channels
        cv2.imshow("distance_1", image_table[0])
        cv2.imshow("intensity_1", image_table[1])
        cv2.imshow("distance_2", image_table[2])
        cv2.imshow("intensity_2", image_table[3])
        cv2.imshow("distance_3", image_table[4])
        cv2.imshow("intensity_3", image_table[5])
        cv2.imshow("label", image_table[6])

        # use color_map to show the label
        label_to_show = np.zeros((pixel_row, pixel_col, 3), dtype=np.uint8)
        for i in range(pixel_row):
            for j in range(pixel_col):
                label_to_show[i, j] = color_map[image_table[6, i, j]]
        cv2.imshow("label", label_to_show)
        cv2.waitKey(0)


def show_test_result_v2():
    predictions = np.load("test_predictions.npy")
    labels = np.load("test_labels.npy")

    print(predictions.shape)
    print(labels.shape)

    origin_id = 1430

    for i in range(predictions.shape[0]):
        id = np.uint32(i) + np.uint32(origin_id)
        # X Y Z row col pluse echo theta phi Original_cloud_index
        frame_array = np.loadtxt(
            "rayz/without_stra_light/{}.txt".format(id)).reshape(-1, 10)
        label_image = predictions[i, :, :]

        for j in range(frame_array.shape[0]):
            tlabel = label_image[int(frame_array[j, 3]), int(frame_array[j, 4])]
            tflag = 2**int(frame_array[j, 6]) & int(tlabel)
            frame_array[j, -1] = tflag

        np.savetxt("rayz/res/{}.txt".format(id), frame_array)


# for i in range(1850, 1860, 10):
#     generate_scan_pattern_image(i)
# verify_hdf5("rayz/train/400.hdf5")
# show_test_result()


def save_as_hdf5(total, save_name):

    image_table = np.zeros((7, 128, 1200))
    # 用3个bit来编码三次回波是否是水雾的情况
    for i in range(128):
        candidate = total[total[:, 3] == i]
        for j in range(1200):

            points = candidate[candidate[:, 4] == j]

            if points.shape[0] != 0:
                first_echo = points[points[:, 6] == 0]
                second_echo = points[points[:, 6] == 1]
                third_echo = points[points[:, 6] == 2]
                label = 0
                if first_echo.shape[0] != 0:
                    image_table[0, i, j] = np.linalg.norm(first_echo[0, :3])
                    image_table[1, i, j] = first_echo[0, 4]
                    if first_echo[0, -1] == 101:
                        label = label | 1
                if second_echo.shape[0] != 0:
                    image_table[2, i, j] = np.linalg.norm(second_echo[0, :3])
                    image_table[3, i, j] = second_echo[0, 4]
                    if second_echo[0, -1] == 101:
                        label = label | 2
                if third_echo.shape[0] != 0:
                    image_table[4, i, j] = np.linalg.norm(third_echo[0, :3])
                    image_table[5, i, j] = third_echo[0, 4]
                    if third_echo[0, -1] == 101:
                        label = label | 4
                image_table[6, i, j] = label

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

    # # use color_map to show the label
    # label_to_show = np.zeros((pixel_row, pixel_col, 3), dtype=np.uint8)
    # for i in range(pixel_row):
    #     for j in range(pixel_col):
    #         label_to_show[i, j] = color_map[image_table[6, i, j]]
    # cv2.imshow("label", label_to_show)
    # cv2.waitKey(0)

    with h5py.File(save_name, "w", driver='core') as hdf5:
        hdf5.create_dataset('distance_1', data=image_table[0].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('intensity_1', data=image_table[1].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('distance_2', data=image_table[2].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('intensity_2', data=image_table[3].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('distance_3', data=image_table[4].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('intensity_3', data=image_table[5].reshape(
            128, 1200), dtype=np.float32)
        hdf5.create_dataset('label', data=image_table[6].reshape(
            128, 1200), dtype=np.float32)


def process():
    # txt -> hdf5

    dir = "/Users/xavier/Documents/calibration/data/txt/shuiwu_8"
    hdf5_dir = "rayz/test"
    origin_dir = "rayz/origin"
    # h_angle, v_angle, range, pluse, echo, row, col
    kRangeResolution = 0.008

    already = os.listdir(hdf5_dir)
    filelist = os.listdir(dir)

    # remove already processed files
    for file in already:
        filelist.remove(file.split(".")[0] + ".txt")

    for filename in filelist:
        id = int(filename.split(".")[0])
        filename = os.path.join(dir, filename)

        txt_data = np.loadtxt(filename).reshape(-1, 7)

        # X Y Z row col pluse echo theta phi Original_cloud_index
        data_to_save = np.zeros((txt_data.shape[0], 10))

        data_to_save[:, 7] = txt_data[:, 1].astype(
            np.int16) / 128.0 * np.pi / 180.0  # theta
        data_to_save[:, 8] = txt_data[:, 0].astype(
            np.int16) * 0.016 * np.pi / 180.0  # phi

        radius = txt_data[:, 2] * kRangeResolution
        t = radius * np.cos(data_to_save[:, 7])

        data_to_save[:, 0] = t * np.sin(data_to_save[:, 8])  # x
        data_to_save[:, 1] = t * np.cos(data_to_save[:, 8])  # y
        data_to_save[:, 2] = radius * np.sin(data_to_save[:, 7])  # z
        data_to_save[:, 3] = txt_data[:, 5]  # row
        data_to_save[:, 4] = txt_data[:, 6]  # col
        data_to_save[:, 5] = txt_data[:, 3].astype(np.int32)  # pluse
        data_to_save[:, 6] = txt_data[:, 4].astype(np.int32) - 1  # echo
        data_to_save[:, 9] = id

        # save original data
        np.savetxt(os.path.join(origin_dir, os.path.basename(filename)), data_to_save)

        # save hdf5 image
        start_time = time.time()
        save_as_hdf5(data_to_save, os.path.join(hdf5_dir, "{}.hdf5".format(id)))
        end_time = time.time()
        print("ID: {}, Time: {}".format(id, end_time - start_time))


def show_test_result():
    origin = "rayz/t8/origin"
    dir = "result"
    filelist = os.listdir(dir)
    
    account = len(filelist)

    for filename in filelist:
        id = int(filename.split(".")[0])
        filename = os.path.join(dir, filename)
        
        res = np.load(filename)

        origin_pcd_name = os.path.join(origin, "{}.txt".format(id))

        frame_array = np.loadtxt(origin_pcd_name).reshape(-1, 10)

        for j in range(frame_array.shape[0]):
            tlabel = res[int(frame_array[j, 3]), int(frame_array[j, 4])]
            tflag = 2**int(frame_array[j, 6]) & int(tlabel)
            frame_array[j, -1] = tflag

        np.savetxt("rayz/res/{}.txt".format(id), frame_array)
        account -= 1
        print ("Remaining: {}".format(account))


def debug_pred():
    pre_data = np.load("result/1560.npy")

    # data = np.load("rayz/test_predictions.npy")
    # res = data[0, :, :]

    good_pred = np.load("rayz/8/test_predictions.npy")
    good_label = good_pred[130, :, :]

    # use color_map to show the label
    predict_to_show = np.zeros((128, 1200, 3), dtype=np.uint8)
    # label_to_show = np.zeros((128, 1200, 3), dtype=np.uint8)
    good_label_to_show = np.zeros((128, 1200, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(1200):
            predict_to_show[i, j] = color_map[pre_data[i, j]]
            # label_to_show[i, j] = color_map[res[i, j]]
            good_label_to_show[i, j] = color_map[good_label[i, j]]
    cv2.imshow("predict", predict_to_show)
    # cv2.imshow("label", label_to_show)
    cv2.imshow("good_label", good_label_to_show)
    cv2.waitKey(0)


# def debug_hdf5():
#     lhv_filename = "rayz/8/selected/train/1440.hdf5"
#     rhv_filename = "rayz/t8/test/1441.hdf5"

# process()
show_test_result()
# debug_pred()
