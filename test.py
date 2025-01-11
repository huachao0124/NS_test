import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('/mnt/search01/usr/xiaosong/zhuhuachao/data/NightCity/NightCity-images/images/val/YouTube_0044.png')  # 请替换为你的图像路径
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算直方图
hist_size = 256
hist_range = [0, 256]
gray_hist = cv2.calcHist([gray_img], [0], None, [hist_size], hist_range)

# 分离RGB通道
R, G, B = cv2.split(img)

# 计算每个通道的直方图
r_hist = cv2.calcHist([R], [0], None, [hist_size], hist_range)
g_hist = cv2.calcHist([G], [0], None, [hist_size], hist_range)
b_hist = cv2.calcHist([B], [0], None, [hist_size], hist_range)

# 在Matplotlib中画图
fig, axs = plt.subplots(4, 2, figsize=(10, 8))

# 显示原图和灰度图以及它们的直方图
axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# axs[0, 1].imshow(gray_img, cmap='gray')
# axs[0, 1].set_title('Gray Image')
# axs[0, 1].axis('off')

# 绘制灰度图直方图
axs[0, 1].plot(gray_hist, color='black')
axs[0, 1].set_xlim([0, 256])
axs[0, 1].set_title('Gray Histogram')

# 显示R通道及其直方图
axs[1, 0].imshow(R, cmap='Reds')
axs[1, 0].set_title('R Channel')
axs[1, 0].axis('off')

axs[1, 1].plot(r_hist, color='red')
axs[1, 1].set_xlim([0, 256])
axs[1, 1].set_title('Red Histogram')

# 显示G通道及其直方图
axs[2, 0].imshow(G, cmap='Greens')
axs[2, 0].set_title('G Channel')
axs[2, 0].axis('off')

axs[2, 1].plot(g_hist, color='green')
axs[2, 1].set_xlim([0, 256])
axs[2, 1].set_title('Green Histogram')

# 显示B通道及其直方图
axs[3, 0].imshow(B, cmap='Blues')
axs[3, 0].set_title('B Channel')
axs[3, 0].axis('off')

axs[3, 1].plot(b_hist, color='blue')
axs[3, 1].set_xlim([0, 256])
axs[3, 1].set_title('Blue Histogram')

plt.tight_layout()
plt.show()
plt.save('test.png')