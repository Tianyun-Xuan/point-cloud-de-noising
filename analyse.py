import numpy as np
import matplotlib.pyplot as plt
# # 1220帧训练 296帧验证

# # all_train no_dropout dropout
# log_1 = "Epoch 1/10, Train Loss: 0.22886433162443015, Train Acc: 0.952343114659838 \nEpoch 1/10, Val Loss: 0.07213394799796136, Val Acc: 0.9807689778182818 \nEpoch 2/10, Train Loss: 0.08144228297245243, Train Acc: 0.9788627012916233 \nEpoch 2/10, Val Loss: 0.06310387244043143, Val Acc: 0.9828567846961643 \nEpoch 3/10, Train Loss: 0.0642192533482676, Train Acc: 0.9828749146150506 \nEpoch 3/10, Val Loss: 0.055664581023966486, Val Acc: 0.9850577870140905 \nEpoch 4/10, Train Loss: 0.056142913278840155, Train Acc: 0.9854714334011078 \nEpoch 4/10, Val Loss: 0.049002304588161086, Val Acc: 0.9876743171526038 \nEpoch 5/10, Train Loss: 0.050407108543035775, Train Acc: 0.987589430809021 \nEpoch 5/10, Val Loss: 0.04383193671946292, Val Acc: 0.9891652143519858 \nEpoch 6/10, Train Loss: 0.045048677427289276, Train Acc: 0.9891016960144043 \nEpoch 6/10, Val Loss: 0.04044122551770314, Val Acc: 0.9902413243832795 \nEpoch 7/10, Train Loss: 0.041606784061245294, Train Acc: 0.9902491077132847 \nEpoch 7/10, Val Loss: 0.03713915483094752, Val Acc: 0.9913007033907849 \nEpoch 8/10, Train Loss: 0.03834779003189634, Train Acc: 0.9912117906238722 \nEpoch 8/10, Val Loss: 0.03515627351387039, Val Acc: 0.9919304772563603 \nEpoch 9/10, Train Loss: 0.037194069472910915, Train Acc: 0.9916664973549221 \nEpoch 9/10, Val Loss: 0.03364296625975682, Val Acc: 0.9923604454683221 \nEpoch 10/10, Train Loss: 0.0355728888924679, Train Acc: 0.9920878402564837 \nEpoch 10/10, Val Loss: 0.03226077704406951, Val Acc: 0.9927182163881219 "
# log_2 = "Epoch 1/10, Train Loss: 0.16436964951944155, Train Acc: 0.9641757228335396 \nEpoch 1/10, Val Loss: 0.2203494392939516, Val Acc: 0.9198110417739765 \nEpoch 2/10, Train Loss: 0.08585260342501226, Train Acc: 0.9759897560369774 \nEpoch 2/10, Val Loss: 0.20253459827319995, Val Acc: 0.9251451371489344 \nEpoch 3/10, Train Loss: 0.06586080180694823, Train Acc: 0.9828975454705661 \nEpoch 3/10, Val Loss: 0.12852215766906738, Val Acc: 0.9643915424475799 \nEpoch 4/10, Train Loss: 0.05852029138534773, Train Acc: 0.9855551346403654 \nEpoch 4/10, Val Loss: 0.12148093945673995, Val Acc: 0.9663786042380977 \nEpoch 5/10, Train Loss: 0.05585416656170712, Train Acc: 0.9863487454711414 \nEpoch 5/10, Val Loss: 0.0902525885282336, Val Acc: 0.9780381434672588 \nEpoch 6/10, Train Loss: 0.051636955520657245, Train Acc: 0.9875069692486622 \nEpoch 6/10, Val Loss: 0.07802347773434343, Val Acc: 0.9810370198778204 \nEpoch 7/10, Train Loss: 0.050308118803335015, Train Acc: 0.9879558397121116 \nEpoch 7/10, Val Loss: 0.0752931420968191, Val Acc: 0.9803317213380659 \nEpoch 8/10, Train Loss: 0.04831238687954477, Train Acc: 0.98853767328575 \nEpoch 8/10, Val Loss: 0.0696411741444388, Val Acc: 0.9829367020645657 \nEpoch 9/10, Train Loss: 0.04694340174681828, Train Acc: 0.9891550478388051 \nEpoch 9/10, Val Loss: 0.07201459646426342, Val Acc: 0.9831228626740945 \nEpoch 10/10, Train Loss: 0.04456426536542226, Train Acc: 0.9899747127392253 \nEpoch 10/10, Val Loss: 0.06564406426371755, Val Acc: 0.9850180124914324"
# log_3 = "Epoch 1/10, Train Loss: 0.13885172689180883, Train Acc: 0.9673530761335717 \nEpoch 1/10, Val Loss: 0.13791400576765472, Val Acc: 0.9526524060481304 \nEpoch 2/10, Train Loss: 0.08005062402760396, Train Acc: 0.9779747042499605 \nEpoch 2/10, Val Loss: 0.16971521492342692, Val Acc: 0.9452466908338908 \nEpoch 3/10, Train Loss: 0.06703609283708158, Train Acc: 0.9815837322688493 \nEpoch 3/10, Val Loss: 0.11286190846884572, Val Acc: 0.9707951465168515 \nEpoch 4/10, Train Loss: 0.057749383948499065, Train Acc: 0.9855507919045745 \nEpoch 4/10, Val Loss: 0.09318485554005648, Val Acc: 0.9760683895768346 \nEpoch 5/10, Train Loss: 0.05320150778002915, Train Acc: 0.9872258256693356 \nEpoch 5/10, Val Loss: 0.08157945796847343, Val Acc: 0.9797049213100124 \nEpoch 6/10, Train Loss: 0.0512862658983127, Train Acc: 0.9879148881943499 \nEpoch 6/10, Val Loss: 0.08102976404935927, Val Acc: 0.9800688849913107 \nEpoch 7/10, Train Loss: 0.04848199160006202, Train Acc: 0.9887698261464228 \nEpoch 7/10, Val Loss: 0.0697779930322557, Val Acc: 0.9824894207554895 \nEpoch 8/10, Train Loss: 0.046621003585150006, Train Acc: 0.9893494393004746 \nEpoch 8/10, Val Loss: 0.06506349720262192, Val Acc: 0.9840163773781544 \nEpoch 9/10, Train Loss: 0.04511290616279499, Train Acc: 0.9897910051658505 \nEpoch 9/10, Val Loss: 0.060040347479485175, Val Acc: 0.9862171210147239 \nEpoch 10/10, Train Loss: 0.04472973973291819, Train Acc: 0.9899237742189502 \nEpoch 10/10, Val Loss: 0.06097477936261409, Val Acc: 0.9870484914328601"

# logs = [log_1, log_2, log_3]
# legend = ["all_train", "no_dropout", "dropout"]

# splited_log = []

# for log in logs:
#     splited_log.append(log.split("\n"))

# epochs = len(splited_log[0]) // 2

# train_loss_table = np.zeros((len(logs), epochs))
# train_acc_table = np.zeros((len(logs), epochs))
# val_loss_table = np.zeros((len(logs), epochs))
# val_acc_table = np.zeros((len(logs), epochs))

# for j in range(len(logs)):
#     log = splited_log[j]
#     for i in range(epochs):
#         train_log = log[2 * i]
#         val_log = log[2 * i + 1]

#         train_loss = train_log.split(",")[1].split(":")[-1]
#         train_acc = train_log.split(",")[2].split(":")[-1]
#         val_loss = val_log.split(",")[1].split(":")[-1]
#         val_acc = val_log.split(",")[2].split(":")[-1]

#         train_loss_table[j][i] = train_loss
#         train_acc_table[j][i] = train_acc
#         val_loss_table[j][i] = val_loss
#         val_acc_table[j][i] = val_acc


# # plot train and val, loss and acc
# plt.figure()
# plt.subplot(2, 1, 1)
# for i in range(len(logs)):
#     plt.plot(train_loss_table[i], label=legend[i] + " train loss")
#     plt.plot(val_loss_table[i], label=legend[i] + " val loss")
# plt.legend()
# plt.title("Loss")
# plt.subplot(2, 1, 2)
# for i in range(len(logs)):
#     plt.plot(train_acc_table[i], label=legend[i] + " train acc")
#     plt.plot(val_acc_table[i], label=legend[i] + " val acc")
# plt.legend()
# plt.title("Acc")
# plt.show()

# 载入txt文本
# 每四行是一个epoche
# 每个epoche 包含如下信息
# Epoch 1/50 - Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 305/305 [12:25<00:00,  2.45s/batch]
# Epoch 1/50, Train Loss: 0.11728162328239347, Train Acc: 0.9733314115493024
# Epoch 1/50 - Validation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 74/74 [01:12<00:00,  1.02batch/s]
# Epoch 1/50, Val Loss: 0.16208648742050738, Val Acc: 0.9348192311622001
# 提取出 每个epoche 的 train loss, train acc, val loss, val acc
# 并画图
log_file = "log.txt"
train_loss = []
train_acc = []
val_loss = []
val_acc = []
account = 0

with open(log_file, "r") as f:
    logs = f.readlines()

    for i in range(len(logs)):
        if i % 4 == 1:
            train_loss.append(float(logs[i].split(",")[1].split(":")[-1]))
            train_acc.append(float(logs[i].split(",")[2].split(":")[-1]))
        if i % 4 == 3:
            val_loss.append(float(logs[i].split(",")[1].split(":")[-1]))
            val_acc.append(float(logs[i].split(",")[2].split(":")[-1]))

epochs = len(train_loss)
x = np.arange(epochs)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, train_loss, label="train loss")
plt.plot(x, val_loss, label="val loss")
plt.legend()
plt.title("Loss")
plt.subplot(2, 1, 2)
plt.plot(x, train_acc, label="train acc")
plt.plot(x, val_acc, label="val acc")
plt.legend()
plt.title("Acc")
plt.show()
