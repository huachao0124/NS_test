import cv2
import numpy as np

img = cv2.imread('/mnt/search01/usr/xiaosong/zhuhuachao/codes/mmsegmentation/data/nightcity-fine/train/img/Chicago_0002.png')
label_map = cv2.imread('/mnt/search01/usr/xiaosong/zhuhuachao/codes/mmsegmentation/data/nightcity-fine/train/lbl/Chicago_0002_trainIds.png', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(label_map.astype(np.uint8), threshold1=100, threshold2=200)
kernel = np.ones((2, 2), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)
# blurred_edges = cv2.GaussianBlur(dilated_edges, (3, 3), 0)
blurred_edges = dilated_edges
# blurred_edges = edges

mask = blurred_edges > 0
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 锐化卷积核
sharpened_img = cv2.filter2D(img, -1, kernel)

enhanced_img = img.copy()
# enhanced_img[mask] = sharpened_img[mask]
enhanced_img[mask] = 255

# enhanced_label_map = cv2.addWeighted(img, 1 - alpha, sharpened_img, alpha, 0)
cv2.imwrite('edges.png', enhanced_img)