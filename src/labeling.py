import os
import numpy as np

# x y z row col range pluse echo （frame_id）[label]

base_dir = "data/mist/pre/jimu"  # 存放原始数据的目录 txt
mark_dir = "data/mist/pre/mark-1"  # 存放选中的水雾数据的目录 txt
save_dir = "data/mist/pre/label-1"  # 存放标记好的数据的目录 txt
npy_dir = "data/mist/pre/npy-1"  # 存放标记好的数据的目录 npy


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
            # print("Error: can't find the point : [{}, {}, {}]".format(
            #     point[3], point[4], point[7]))
            continue
        # 标记为水雾
        basepoints[target_place, 8] = 1
    return basepoints


def markMistPoints(base_dir, mark_dir, save_dir):
    base_file = os.listdir(base_dir)
    # 排序
    base_file.sort()
    mark_file = os.listdir(mark_dir)
    mark_file.sort()
    # remove .DS_Store

    for file in mark_file:
        if file == ".DS_Store":
            continue
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
    files = os.listdir(dir)
    for file in files:
        if "_" in file:
            new_file = file.split("_")[0] + ".txt"
            os.rename(os.path.join(dir, file), os.path.join(dir, new_file))


def generateNpyFile(label_dir, npy_dir):
    files = os.listdir(label_dir)
    for file in files:
        points = np.loadtxt(os.path.join(label_dir, file)).reshape(-1, 9)

        first_echo_x = np.zeros((128, 1200))
        first_echo_y = np.zeros((128, 1200))
        first_echo_z = np.zeros((128, 1200))
        first_echo_range = np.zeros((128, 1200))
        first_echo_pluse = np.zeros((128, 1200))
        second_echo_x = np.zeros((128, 1200))
        second_echo_y = np.zeros((128, 1200))
        second_echo_z = np.zeros((128, 1200))
        second_echo_range = np.zeros((128, 1200))
        second_echo_pluse = np.zeros((128, 1200))
        label_map = np.zeros((128, 1200))

        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
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
                first_echo_x[row, col] = x
                first_echo_y[row, col] = y
                first_echo_z[row, col] = z
            elif echo == 1:
                second_echo_range[row, col] = range
                second_echo_pluse[row, col] = pluse
                second_echo_x[row, col] = x
                second_echo_y[row, col] = y
                second_echo_z[row, col] = z
            if flag == 1:
                label_map[row, col] = int(label_map[row, col]) | mark

        # print("First echo range : [{}, {}]".format(
        #     np.min(first_echo_range), np.max(first_echo_range)))
        # print("First echo pluse : [{}, {}]".format(
        #     np.min(first_echo_pluse), np.max(first_echo_pluse)))
        # print("Second echo range : [{}, {}]".format(
        #     np.min(second_echo_range), np.max(second_echo_range)))
        # print("Second echo pluse : [{}, {}]".format(
        #     np.min(second_echo_pluse), np.max(second_echo_pluse)))

        # 拼接
        data = np.stack(
            (first_echo_x, first_echo_y, first_echo_z, first_echo_range, first_echo_pluse,
             second_echo_x, second_echo_y, second_echo_z, second_echo_range, second_echo_pluse, label_map))
        print(data.shape)
        np.save(os.path.join(npy_dir, file.split(".")[0] + ".npy"), data)


def checkNpyFile(base_dir, npy_dir, check_dir):
    # 加载原始点云和npy中的标注，生成标注点云
    baseFiles = os.listdir(base_dir)
    npyFiles = os.listdir(npy_dir)

    for file in npyFiles:
        id = file.split(".")[0]
        baseFileName = os.path.join(base_dir, id + ".txt")
        npyFileName = os.path.join(npy_dir, file)
        if not os.path.exists(baseFileName):
            print("Error: can't find the file : {}".format(baseFileName))
            continue
        if not os.path.exists(npyFileName):
            print("Error: can't find the file : {}".format(npyFileName))
            continue

        basePoints = np.loadtxt(baseFileName).reshape(-1, 8)
        # 拓展一列用于标记水雾
        basePoints = np.hstack(
            (basePoints, np.zeros((basePoints.shape[0], 1))))
        npyData = np.load(npyFileName).reshape(5, 128, 1200)

        label = npyData[4]

        for point in basePoints:
            row = int(point[3])
            col = int(point[4])
            echo = int(point[7])
            mark = int(2 ** echo)
            if int(label[row, col]) & mark:
                point[8] = 1

        saveFileName = os.path.join(check_dir, id + ".txt")
        np.savetxt(saveFileName, basePoints, fmt="%.6f")


def Dep_checkPredictionFile(label_dir, pred_dir, check_dir):
    # 加载原始点云和npy中的标注，生成标注点云
    labelFiles = os.listdir(label_dir)
    predFiles = os.listdir(pred_dir)
    labelFiles.sort()
    predFiles.sort()

    for i in range(len(labelFiles)):
        labelFileName = os.path.join(label_dir, labelFiles[i])
        predFileName = os.path.join(pred_dir, predFiles[i])
        if not os.path.exists(labelFileName):
            print("Error: can't find the file : {}".format(labelFileName))
            continue
        if not os.path.exists(predFileName):
            print("Error: can't find the file : {}".format(predFileName))
            continue

        basePoints = np.loadtxt(labelFileName).reshape(-1, 9)
        basePoints[:, 8] = 0
        # # 拓展一列用于标记水雾
        # basePoints = np.hstack(
        #     (basePoints, np.zeros((basePoints.shape[0], 1))))
        label = np.load(predFileName).reshape(128, 1200)

        for point in basePoints:
            row = int(point[3])
            col = int(point[4])
            echo = int(point[7])
            if echo == 0:
                mark = int(2 ** echo)
                if int(label[row, col]) & mark:
                    point[8] = 1

        saveFileName = os.path.join(check_dir, labelFiles[i])
        np.savetxt(saveFileName, basePoints, fmt="%.6f")


def regroupeNpyfile(src_dir, dst_dir, mark):
    file = os.listdir(src_dir)
    # copy all file as "[mark]-[file]" to dst_dir
    for f in file:
        os.rename(os.path.join(src_dir, f), os.path.join(
            dst_dir, str(mark) + "-" + f))


def findBaseFile(base_dir_list, npy_dir, dst_dir):
    npyFiles = os.listdir(npy_dir)
    for file in npyFiles:
        datasetID = int(file.split("-")[0]) - 1
        frameID = int(file.split("-")[1].split(".")[0])
        baseFileName = os.path.join(
            base_dir_list[datasetID], str(frameID) + ".txt")
        if not os.path.exists(baseFileName):
            print("Error: can't find the file : {}".format(baseFileName))
            continue

        dstFileName = os.path.join(
            dst_dir, "{}-{}.txt".format(datasetID + 1, frameID))
        # copy file
        os.system("cp {} {}".format(baseFileName, dstFileName))


def reOrderSpliteTxtFile(src_dir, dst_dir):
    files = os.listdir(src_dir)
    files.sort()
    frame_count = 0
    for file in files:
        if file == ".DS_Store":
            continue
        # copy file to dst and save as 00000.txt
        dstFileName = os.path.join(dst_dir, "{:05d}.txt".format(frame_count))
        os.system("cp {} {}".format(os.path.join(src_dir, file), dstFileName))
        frame_count += 1


def reOrderSpliteNpyFile(src_dir, dst_dir, datasetID):
    files = os.listdir(src_dir)
    files.sort()
    frame_count = 0
    for file in files:
        if file == ".DS_Store":
            continue
        # copy file to dst and save as datasetID0000.txt
        # for example: 10000.txt
        dstFileName = os.path.join(
            dst_dir, "{}{:04d}.npy".format(datasetID, frame_count))
        os.system("cp {} {}".format(os.path.join(src_dir, file), dstFileName))
        frame_count += 1


def generateOldNpyFile(label_dir, npy_dir):
    files = os.listdir(label_dir)
    for file in files:
        if file == ".DS_Store":
            continue
        points = np.loadtxt(os.path.join(label_dir, file)).reshape(-1, 10)

        first_echo_x = np.zeros((128, 1200))
        first_echo_y = np.zeros((128, 1200))
        first_echo_z = np.zeros((128, 1200))
        first_echo_range = np.zeros((128, 1200))
        first_echo_pluse = np.zeros((128, 1200))
        second_echo_x = np.zeros((128, 1200))
        second_echo_y = np.zeros((128, 1200))
        second_echo_z = np.zeros((128, 1200))
        second_echo_range = np.zeros((128, 1200))
        second_echo_pluse = np.zeros((128, 1200))
        label_map = np.zeros((128, 1200))

        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            range = int(point[3])
            pluse = int(point[4])
            row = int(point[5])
            col = int(point[6])
            index = int(point[7])
            echo = int(point[8])
            # previous version
            if echo == 2:
                echo = 0
            flag = int(point[9])
            mark = int(2 ** echo)
            if echo == 0:
                first_echo_range[row, col] = range
                first_echo_pluse[row, col] = pluse
                first_echo_x[row, col] = x
                first_echo_y[row, col] = y
                first_echo_z[row, col] = z
            elif echo == 1:
                second_echo_range[row, col] = range
                second_echo_pluse[row, col] = pluse
                second_echo_x[row, col] = x
                second_echo_y[row, col] = y
                second_echo_z[row, col] = z
            if flag == 1:
                label_map[row, col] = int(label_map[row, col]) | mark

        # print("First echo range : [{}, {}]".format(
        #     np.min(first_echo_range), np.max(first_echo_range)))
        # print("First echo pluse : [{}, {}]".format(
        #     np.min(first_echo_pluse), np.max(first_echo_pluse)))
        # print("Second echo range : [{}, {}]".format(
        #     np.min(second_echo_range), np.max(second_echo_range)))
        # print("Second echo pluse : [{}, {}]".format(
        #     np.min(second_echo_pluse), np.max(second_echo_pluse)))

        # 拼接
        data = np.stack(
            (first_echo_x, first_echo_y, first_echo_z, first_echo_range, first_echo_pluse,
             second_echo_x, second_echo_y, second_echo_z, second_echo_range, second_echo_pluse, label_map))
        print(data.shape)
        np.save(os.path.join(npy_dir, file.split(".")[0] + ".npy"), data)


def checkPredictionFile(npy_dir, pred_dir, check_dir):
    # 加载原始点云和npy中的标注，生成标注点云
    npyFiles = os.listdir(npy_dir)
    predFiles = os.listdir(pred_dir)
    # npyFiles.sort()
    # predFiles.sort()

    for file in npyFiles:
        id = file.split(".")[0]
        labelFileName = os.path.join(npy_dir, "{}.npy".format(id))
        predFileName = os.path.join(pred_dir, "{}.npy".format(id))

        if not os.path.exists(labelFileName):
            print("Error: can't find the file : {}".format(labelFileName))
            continue
        if not os.path.exists(predFileName):
            print("Error: can't find the file : {}".format(predFileName))
            continue

        # [x y z range pluse] [x y z range pluse] [label]
        depth = np.load(labelFileName).reshape(11, 128, 1200)
        label = np.load(predFileName).reshape(128, 1200)

        # x y z echo label
        result = []

        for row in range(128):
            for col in range(1200):
                range_1 = depth[3, row, col]
                range_2 = depth[8, row, col]
                if range_1 > 0:
                    x = depth[0, row, col]
                    y = depth[1, row, col]
                    z = depth[2, row, col]
                    pluse = depth[4, row, col]
                    label_1 = label[row, col]
                    result.append([x, y, z, range_1, pluse, 0, label_1])
                if range_2 > 0:
                    x = depth[5, row, col]
                    y = depth[6, row, col]
                    z = depth[7, row, col]
                    pluse = depth[9, row, col]
                    result.append([x, y, z, range_2, pluse, 1, 0])

        result = np.array(result).reshape(-1, 5)
        np.savetxt(os.path.join(check_dir, "{}.txt".format(id)),
                   result, fmt="%.6f")


# generateOldNpyFile("data/8/label", "data/8/npy")
# markMistPoints(base_dir, mark_dir, save_dir)
# reOrderSpliteFile("data/mist/label", "data/mist/src")
# generateNpyFile("data/mist/src", "data/mist/npy")
# checkNpyFile(base_dir, npy_dir, "data/mist/check-1")
# checkPredictionFile("data/mist/src", "data/pred", "data/mist/check")
# reOrderSpliteNpyFile("data/5/npy", "data/5/gnpy", 5)
