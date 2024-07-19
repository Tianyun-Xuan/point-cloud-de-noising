import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(
    "/Users/xavier/Documents/calibration/point-cloud-de-noising/log.txt").reshape(-1, 3)

# total_time infer_time removed_points
total_time = data[:, 0]
infer_time = data[:, 1]
removed_points = data[:, 2]

# 画出3张图
# 在图片旁边如何标注出mean和std？

plt.show()
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle('Delay Analysis')
axs[0].plot(total_time, label='total_time')
axs[0].axhline(y=total_time.mean(), color='r', linestyle='-', label='mean')
axs[0].set_xlabel("frame")
axs[0].set_ylabel("time (ms)")
axs[0].legend(loc='upper right')

axs[1].plot(infer_time, label='infer_time')
axs[1].axhline(y=infer_time.mean(), color='r', linestyle='-', label='mean')
axs[1].set_xlabel("frame")
axs[1].set_ylabel("time (ms)")
axs[1].legend(loc='upper right')

axs[2].plot(removed_points, label='removed_points')
axs[2].axhline(y=removed_points.mean(), color='r', linestyle='-', label='mean')
axs[2].set_xlabel("frame")
axs[2].set_ylabel("points")
axs[2].legend(loc='upper right')

plt.savefig("delay_analysis.png")
plt.close()
