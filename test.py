import numpy as np
import os

file_list = os.listdir("data/train")
unique_list = np.array([])

for file in file_list:
    data = np.load("data/train/" + file).reshape(128,1200,5)
    label = np.unique(data[:,:,4])
    unique_list = np.unique(np.concatenate((unique_list, label)))

unique_list = unique_list.astype(int)
print(unique_list)