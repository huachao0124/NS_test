import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

def calculate_average_gray(folder_path):
    """
    计算文件夹内所有图像的平均灰度值
    """
    
    # 存储所有图像的平均灰度值
    gray_values = []
    
    # 处理每张图像
    for image_path in tqdm(glob.glob(folder_path), desc="处理图像"):
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)
        
        if img is not None:
            # 转换为灰度图并计算平均值
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_gray = np.mean(gray)
            gray_values.append((filename, mean_gray))
    
    # 计算总体平均值
    overall_mean = np.mean([v[1] for v in gray_values])
    
    # 打印结果
    print(f"\n总体平均灰度值: {overall_mean:.2f}")
    # print("\n各图像的平均灰度值:")
    # for filename, value in gray_values:
    #     print(f"{filename}: {value:.2f}")
    
    return overall_mean, gray_values

# 使用示例
folder_path = "data/cityscapes/leftImg8bit/**/**/*.png"  # 替换为你的图像文件夹路径
calculate_average_gray(folder_path)