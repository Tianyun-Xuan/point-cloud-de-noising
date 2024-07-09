# 使用 SVM来进行水雾的一分类任务

# 输入数据 行号 列号 距离 脉宽
# 输出 1 代表水雾 0 代表非水雾

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import tqdm

# 我的数据很多分布在多个文件中，我需要先划分数据集

# 读取数据


def load_data(data_path):
    # x y z range pluse row col index echo label
    data = np.loadtxt(data_path).reshape(-1, 10)

    # echo == 2 and label == 1
    useful_data = data[(data[:, 8] == 2) & (data[:, 9] == 1)]

    # range pluse row col
    input = useful_data[:, 3:7]

    return input


# 数据文件
files = os.listdir('data/5/label/')

# 划分训练和测试集
train_files = files[:int(len(files)*0.8)]
test_files = files[int(len(files)*0.8):]

# 建立单分类SVM模型，采用RBF 核函数
model = svm.OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)

# 训练模型
for file in tqdm.tqdm(train_files):
    data = load_data('data/5/label/' + file)
    model.fit(data)

# 测试模型
y_true = []
y_pred = []
for file in tqdm.tqdm(test_files):
    data = load_data('data/5/label/' + file)
    y_true += [1]*len(data)
    y_pred += list(model.predict(data))

# 计算准确率和召回率
accuracy = accuracy_score(y_true, y_pred)
print('accuracy:', accuracy)
confusion = confusion_matrix(y_true, y_pred)
print('confusion matrix:')
print(confusion)
print('classification report:')
print(classification_report(y_true, y_pred))

# 画图
plt.figure()
plt.plot(y_true, label='true')
plt.plot(y_pred, label='predict')
plt.legend()

plt.show()
