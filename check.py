# 检查标注的深度图还原点云的效果

import numpy as np

print(int(pow(2, 2- 2)))
print(int(pow(2, 2- 1)))

# x y z range pluse row col index echo label
point_label = np.loadtxt("data/8/label/27628.txt").reshape(-1, 10)
depth_map = np.load("data/8/npy/27628.npy").reshape(5, 128, 1200)

labels = depth_map[4, :, :]

# 遍历点云 echo是 2 或者 1 的检查标注是否一致
for i in range(0, point_label.shape[0]):
    echo = int(point_label[i, 8])
    flag = int(point_label[i, -1])  # 0 or 1
    if echo == 2 or echo == 1:
        row = int(point_label[i, 5])
        col = int(point_label[i, 6])
        label = int(labels[row, col])

        value = int(pow(2, 2- echo))

        if value & label:
            point_label[i, 4] = 1
        else:
            point_label[i, 4] = 0

        if point_label[i, 4] != flag:
            print("error")
            print(i)
            print(point_label[i, :])
            print(row, col, label)
            print(echo)
            print(value)
            print(label)
            print(flag)

np.savetxt("data/test.txt", point_label, fmt="%f")
