import os
import numpy as np

# x y z row col range pluse echo （frame_id）[label]

base_dir = "data/mist/jimu-2"  # 存放原始数据的目录 txt
mark_dir = "data/mist/mark-2"  # 存放选中的水雾数据的目录 txt
save_dir = "data/mist/label-2"  # 存放标记好的数据的目录 txt
npy_dir = "data/mist/npy-2"  # 存放标记好的数据的目录 npy


def markMistPoint(mistpoints, basepoints):
    # 计算每个点的index = row * 1200 + col + 128* 1200 * echo
    index = basepoints[:, 3] * 1200 + \
        basepoints[:, 4] + 128 * 1200 * basepoints[:, 7]
    index = index.astype(int)
    for point in mistpoints:
        # 通过 row col echo 找到对应的点
        target_index = int(point[3] * 1200 +
                           point[4] + 128 * 1200 * point[7])
        target_place = np.where(index == target_index)
        if len(target_place[0]) == 0:
            print("Error: can't find the point : [{}, {}, {}]".format(
                point[3], point[4], point[7]))
            continue
        # 标记为水雾
        basepoints[target_place, 8] = 1
    return basepoints


def markMistPoint(base_dir, mark_dir, save_dir, npy_dir):
    base_file = os.listdir(base_dir)
    # 排序
    base_file.sort()
    mark_file = os.listdir(mark_dir)
    mark_file.sort()

    for file in mark_file:
        # 读取文件的第一行，以空格为分界，看第一行数字的个数
        fsize = 0
        with open(os.path.join(mark_dir, file), 'r') as f:
            firstLine = f.readline()
            fsize = len(firstLine.split(" "))

        if fsize == 8:
            markFilename = os.path.join(mark_dir, file)
            baseFilename = os.path.join(base_dir, file)
            saveFilename = os.path.join(save_dir, file)

            # 读取数据
            if not os.path.exists(baseFilename):
                print("Error: can't find the file : {}".format(baseFilename))
                continue
            if not os.path.exists(markFilename):
                print("Error: can't find the file : {}".format(baseFilename))
                continue
            mistPoints = np.loadtxt(markFilename).reshape(-1, 8)
            basePoints = np.loadtxt(baseFilename).reshape(-1, 8)
            # 拓展一列用于标记水雾
            basePoints = np.hstack(
                (basePoints, np.zeros((basePoints.shape[0], 1))))
            # 标记水雾
            savePoints = markMistPoint(mistPoints, basePoints)
            # 保存数据
            np.savetxt(saveFilename, savePoints, fmt="%.6f")
        elif fsize == 9:
            startIndex = int(file.split(".")[0])
            dataSets = np.loadtxt(os.path.join(mark_dir, file)).reshape(-1, 9)
            setIds = np.unique(dataSets[:, 8]).astype(int)
            setIds.sort()

            for id in setIds:
                fileIndex = int(startIndex + id)
                fileName = str(fileIndex) + ".txt"

                baseFilename = os.path.join(base_dir, fileName)
                saveFilename = os.path.join(save_dir, fileName)

                if not os.path.exists(baseFilename):
                    print("Error: can't find the file : {}".format(baseFilename))
                    continue
                mistPoints = dataSets[dataSets[:, 8] == id][:, :8]
                basePoints = np.loadtxt(baseFilename).reshape(-1, 8)
                # 拓展一列用于标记水雾
                basePoints = np.hstack(
                    (basePoints, np.zeros((basePoints.shape[0], 1))))
                # 标记水雾
                savePoints = markMistPoint(mistPoints, basePoints)
                # 保存数据
                np.savetxt(saveFilename, savePoints, fmt="%.6f")

        else:
            print("Error: the file is not in the right format : {} has filesize {}".format(
                file, fsize))


def renameLabelFile(dir):
    # rename "7187 - Cloud.txt" to "7187.txt"
    files = os.listdir(dir)
    for file in files:
        if " - " in file:
            new_file = file.split(" - ")[0] + ".txt"
            os.rename(os.path.join(dir, file), os.path.join(dir, new_file))


def remove_DS_Store(dir):
    files = os.listdir(dir)
    for file in files:
        if ".DS_Store" in file:
            os.remove(os.path.join(dir, file))


def generateNpyFile(label_dir, npy_dir):
    files = os.listdir(label_dir)
    for file in files:
        points = np.loadtxt(os.path.join(label_dir, file)).reshape(-1, 9)

        first_echo_range = np.zeros((128, 1200))
        first_echo_pluse = np.zeros((128, 1200))
        second_echo_range = np.zeros((128, 1200))
        second_echo_pluse = np.zeros((128, 1200))
        label_map = np.zeros((128, 1200))

        for point in points:
            row = int(point[3])
            col = int(point[4])
            range = int(point[5])
            pluse = int(point[6])
            echo = int(point[7])
            flag = int(point[8])
            mark = int(2 ** echo)
            if echo == 0:
                first_echo_range[row, col] = range
                first_echo_pluse[row, col] = pluse
            elif echo == 1:
                second_echo_range[row, col] = range
                second_echo_pluse[row, col] = pluse
            if flag == 1:
                label_map[row, col] = int(label_map[row, col]) | mark

        # 拼接成(5, 128, 1200)
        data = np.stack(
            (first_echo_range, first_echo_pluse, second_echo_range, second_echo_pluse, label_map))
        print(data.shape)
        np.save(os.path.join(npy_dir, file.split(".")[0] + ".npy"), data)

generateNpyFile(save_dir, npy_dir)

# def checkNpyFile(base_dir, npy_dir, check_dir):
#     # 加载原始点云和npy中的标注，生成标注点云
#     baseFiles = os.listdir(base_dir)
#     npyFiles = os.listdir(npy_dir)

#     for file in npyFiles:
#         id = file.split(".")[0]
#         baseFileName = os.path.join(base_dir, id + ".txt")
#         npyFileName = os.path.join(npy_dir, file)
#         if not os.path.exists(baseFileName):
#             print("Error: can't find the file : {}".format(baseFileName))
#             continue
#         if not os.path.exists(npyFileName):
#             print("Error: can't find the file : {}".format(npyFileName))
#             continue

#         basePoints = np.loadtxt(baseFileName).reshape(-1, 8)
#         # 拓展一列用于标记水雾
#         basePoints = np.hstack(
#             (basePoints, np.zeros((basePoints.shape[0], 1))))
#         npyData = np.load(npyFileName).reshape(5, 128, 1200)

#         label = npyData[4]

#         for point in basePoints:
#             row = int(point[3])
#             col = int(point[4])
#             echo = int(point[7])
#             mark = int(2 ** echo)
#             if int(label[row, col]) & mark:
#                 point[8] = 1

#         saveFileName = os.path.join(check_dir, id + ".txt")
#         np.savetxt(saveFileName, basePoints, fmt="%.6f")

# checkNpyFile(base_dir, npy_dir, "data/mist/check-2")
