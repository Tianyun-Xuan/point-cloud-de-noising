from dataset import create_dataloader, NPYDataset
import matplotlib.pyplot as plt
import numpy as np
data_loader = create_dataloader('data/8/npy', 1)

# 获取数据，统计其中label出现的频次，生成直方图，说明标注数据样本不均衡
label_count = {0: 0, 1: 0, 2: 0, 3: 0}
for batch in data_loader:
    data, labels = batch
    # 拉平labels
    labels = labels.view(-1)
    label_0 = labels[labels == 0].size(0)
    label_1 = labels[labels == 1].size(0)
    label_2 = labels[labels == 2].size(0)
    label_3 = labels[labels == 3].size(0)
    label_count[0] += label_0
    label_count[1] += label_1
    label_count[2] += label_2
    label_count[3] += label_3
print(label_count)

# 在每个bar上显示具体数值，每个bar用不同的颜色
plt.bar(label_count.keys(), label_count.values(), color=['red', 'green', 'blue', 'yellow'])
for a, b in zip(list(label_count.keys()), list(label_count.values())):
    plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=20)
# 设置x轴标签
plt.xticks(list(label_count.keys()),fontsize=20)
plt.title('Label distribution', fontsize=20)
plt.xlabel('Label', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.show()

