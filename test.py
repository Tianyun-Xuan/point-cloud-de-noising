import numpy as np
import os

# file_list = os.listdir("data/train")
# save_dir = "data/c"
# for file in file_list:
#     data = np.load("data/train/" + file).reshape(5,128,1200)
#     data = data.reshape(5, 128 * 1200)
#     np.savetxt(os.path.join(save_dir, file.replace("npy", "txt")), data, fmt="%5f")


# data = np.load("/home/rayz/code/data/class8/8/train/1477.npy").reshape(7,128,1200)

# data = np.load("data/8/npy/27628.npy").reshape(5, 128, 1200)
# annotation = data[4, :, :]
# result = np.load("data/infer_result_4.npy").reshape(4, 128, 1200)

# # count non-zero
# non_zero_annotation = np.count_nonzero(annotation)
# non_zero_result = np.count_nonzero(result)

# calculate precison for each class, we have 4 classes 0 1 2 3
# calculate recall for each class
# precision = np.zeros(4)
# recall = np.zeros(4)

# for i in range(4):
#     precision[i] = np.sum(np.logical_and(
#         annotation == i, result == i)) / np.sum(result == i)
#     recall[i] = np.sum(np.logical_and(
#         annotation == i, result == i)) / np.sum(annotation == i)

# print(precision)
# print(recall)


# dim_4 = []

# for i in range(128):
#     for j in range(1200):
#         if annotation[i][j] == 1:
#             dim_4.append(result[:, i, j])

# dim_4 = np.array(dim_4).reshape(-1, 4) 

import torch

print(torch.__version__)