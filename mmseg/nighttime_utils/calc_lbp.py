import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

# def calc_lbp(image):
#     lbp = local_binary_pattern(image, P=8, R=1, method='default')
#     print(lbp.shape)
#     hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
#     print(hist.shape)
#     return lbp, hist

# 噪声抗性 LBP
def nrlbp(image, P=8, R=1, epsilon=0.01):
    rows, cols = image.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(R, rows-R):
        for j in range(R, cols-R):
            center = image[i,j]
            code = 0
            for k in range(P):
                x = i + R * np.cos(2*np.pi*k/P)
                y = j - R * np.sin(2*np.pi*k/P)
                x, y = int(x), int(y)
                if abs(int(image[x,y]) - int(center)) <= epsilon:
                    bit = 0
                elif image[x,y] > center:
                    bit = 1
                else:
                    bit = 0
                code |= (bit << k)
            lbp[i,j] = code  # 将计算的code赋值给result
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return lbp, hist


img = cv2.imread('/mnt/nj-public02/usr/xiangyiwei/zhuhuachao/NS_test/data/nightcity-fine/train/img/Chicago_0005.png')  # 读取图像文件
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(img_rgb)
ax2.imshow(img_hsv[:, :, 2])

lbp, hist = nrlbp(img_hsv[:, :, 2])

ax3.imshow(lbp)
ax4.plot(hist)



plt.savefig('test_nrlbp.png')
