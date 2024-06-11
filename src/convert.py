import os
import numpy as np

input_dir = 'data/class8/train'
output_dir = 'data/train'

file_list = os.listdir(input_dir)

# 原先的数据是 7 个通道，3组距离和脉宽，一个标签通道
# 现在去掉第三组距离和脉宽，只保留前两组，并且标签通道从8类(0-7)变为4类(0-3)

for file in file_list:
    data = np.load(os.path.join(input_dir, file)).reshape(128, 1200, 7)
    image = data[:, :, :4]
    label = data[:, :, 6]

    label = np.array(label, dtype=np.int32).reshape(128,1200,1)
    label = label & 3

    image = image.reshape(128, 1200, 4)
    label = label.reshape(128, 1200, 1)
    
    data = np.concatenate((image, label), axis=2)
    
    print (data.shape)
    np.save(os.path.join(output_dir, file), data)
