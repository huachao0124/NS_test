import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import glob


def calculate_multiple_night_histograms(night_folder):
    """计算并保存所有夜景图像的直方图"""
    import os
    
    histograms = []
    
    for filename in os.listdir(night_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(night_folder, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                # 转换为YUV并获取Y通道
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                y_channel = yuv[:,:,0]
                
                # 计算直方图
                hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
                histograms.append(hist.flatten())
    
    return np.array(histograms)

def random_histogram_matching(source_image, reference_hists, variation_range=0.2):
    """带随机变化的直方图匹配"""
    # 随机选择一个参考直方图
    reference_hist = random.choice(reference_hists)
    
    # 添加随机变化
    variation = np.random.uniform(1-variation_range, 1+variation_range, 256)
    reference_hist = reference_hist * variation
    reference_hist = reference_hist / reference_hist.sum() * reference_hist.size
    
    # 计算源图像的直方图
    source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    
    # 计算累积分布函数 (CDF)
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
    
    # 归一化CDF
    source_cdf_normalized = source_cdf / source_cdf.max()
    reference_cdf_normalized = reference_cdf / reference_cdf.max()
    
    # 创建查找表
    lookup_table = np.zeros(256)
    for i in range(256):
        j = 255
        while j >= 0 and reference_cdf_normalized[j] > source_cdf_normalized[i]:
            j -= 1
        lookup_table[i] = j
    
    return cv2.LUT(source_image, lookup_table.astype('uint8'))

def adjust_color_temperature(image, adjustment=0.8):
    """调整色温"""
    b, g, r = cv2.split(image)
    r = cv2.multiply(r, adjustment)
    return cv2.merge([b, g, r])

def process_day_to_night(day_image_path, night_hists, output_path=None, show_result=True):
    """将白天图像转换为夜景风格"""
    # 读取白天图像
    day_image = cv2.imread(day_image_path)
    if day_image is None:
        raise Exception("Could not read the day image")
    
    # 转换为YUV色彩空间
    day_yuv = cv2.cvtColor(day_image, cv2.COLOR_BGR2YUV)
    
    # 对Y通道进行直方图匹配
    day_yuv[:,:,0] = random_histogram_matching(day_yuv[:,:,0], night_hists)
    
    # 转换回BGR
    result = cv2.cvtColor(day_yuv, cv2.COLOR_YUV2BGR)
    
    # 调整色温和饱和度
    result = adjust_color_temperature(result)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * np.random.uniform(0.7, 0.9)  # 随机调整饱和度
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # # 添加轻微的噪点
    # noise = np.random.normal(0, 2, result.shape).astype(np.uint8)
    # result = cv2.add(result, noise)
    
    if show_result:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(day_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Day Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Night Style Result')
        plt.axis('off')
        
        plt.subplot(133)
        plt.plot(cv2.calcHist([cv2.cvtColor(day_image, cv2.COLOR_BGR2GRAY)],
                             [0], None, [256], [0, 256]), 'g', label='Original')
        plt.plot(cv2.calcHist([cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)],
                             [0], None, [256], [0, 256]), 'r', label='Result')
        plt.title('Histograms')
        plt.legend()
        plt.show()
    
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

# 批量处理示例
def batch_process(day_folder, night_folder, output_folder):
    """批量处理白天图像"""
    import os
    from tqdm import tqdm
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有夜景直方图
    night_hists = calculate_multiple_night_histograms(night_folder)
    
    # 处理所有白天图像
    for filename in tqdm(glob.glob(day_folder)):
        # input_path = os.path.join(day_folder, filename)
        # output_path = os.path.join(output_folder, f"night_{filename}")
        output_path = day_folder.replace('cityscapes', 'cityscapes_night')
        folder_path = os.path.dirname(output_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        try:
            process_day_to_night(
                input_path,
                night_hists,
                output_path=output_path,
                show_result=False
            )
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    day_folder = "data/cityscapes/leftImg8bit/train/**/*.png"
    night_folder = "data/nightcity-fine/train/img"
    output_folder = "data/cityscapes_night/leftImg8bit/train"
    
    batch_process(day_folder, night_folder, output_folder)
