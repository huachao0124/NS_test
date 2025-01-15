import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

from concurrent.futures import ThreadPoolExecutor
import threading

class DayToNight:
    def __init__(self):
        self.night_features = None
        
    def load_images(self, day_path, night_path):
        """
        加载日景和夜景图像
        """
        # 加载日景图像
        self.day_images = []
        self.day_image_paths = []
        for img_path in glob.glob(day_path):
            img = cv2.imread(img_path)
            if img is not None:
                self.day_images.append(img)
                self.day_image_paths.append(img_path)
        
        # 加载夜景图像
        self.night_images = []
        for img_path in glob.glob(night_path):
            img = cv2.imread(img_path)
            if img is not None:
                # 调整图像大小以加快处理速度
                img = cv2.resize(img, (2048, 1024))
                self.night_images.append(img)
                    
        print(f"已加载 {len(self.day_images)} 张日景图像和 {len(self.night_images)} 张夜景图像")
        
    def histogram_matching(self, source, reference):
        """
        直方图匹配
        """
        # 转换到LAB色彩空间
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        matched = source_lab.copy()
        for i in range(3):
            # 计算累积分布函数
            src_hist, _ = np.histogram(source_lab[:,:,i].ravel(), 256, [0,256])
            ref_hist, _ = np.histogram(reference_lab[:,:,i].ravel(), 256, [0,256])
            
            src_cdf = src_hist.cumsum()
            ref_cdf = ref_hist.cumsum()
            
            src_cdf = src_cdf / src_cdf[-1]
            ref_cdf = ref_cdf / ref_cdf[-1]
            
            # 创建查找表
            lookup_table = np.zeros(256)
            j = 0
            for idx in range(256):
                while j < 256 and ref_cdf[j] <= src_cdf[idx]:
                    j += 1
                lookup_table[idx] = j
            
            matched[:,:,i] = cv2.LUT(source_lab[:,:,i], lookup_table)
        
        # 转换回BGR色彩空间
        result = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        return result
    
    def color_transfer(self, source, reference):
        """
        颜色迁移
        """
        # 转换到LAB色彩空间
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(float)
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(float)
        
        # 计算均值和标准差
        source_mean = np.mean(source, axis=(0,1))
        source_std = np.std(source, axis=(0,1))
        reference_mean = np.mean(reference, axis=(0,1))
        reference_std = np.std(reference, axis=(0,1))
        
        # 应用变换
        result = source.copy()
        for i in range(3):
            result[:,:,i] = ((source[:,:,i] - source_mean[i]) * 
                           (reference_std[i] / source_std[i])) + reference_mean[i]
        
        # 裁剪值到有效范围
        result = np.clip(result, 0, 255)
        
        # 转换回BGR色彩空间
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
    
    def post_process(self, img):
        """
        后处理：调整亮度、对比度等
        """
        # 降低亮度
        alpha = 0.85
        beta = -10
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 增加蓝色调
        b, g, r = cv2.split(img)
        b = cv2.add(b, 10)
        img = cv2.merge([b, g, r])
        
        # 降噪
        # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        return img
    
    def process_single_image(self, args):
        """
        处理单张图像
        """
        i, day_img, day_img_path = args
        
        # 随机选择一张夜景图像
        night_idx = np.random.randint(len(self.night_images))
        night_img = self.night_images[night_idx]
        
        # 直方图匹配
        result1 = self.histogram_matching(day_img, night_img)
        
        # 颜色迁移
        result2 = self.color_transfer(day_img, night_img)
        
        # 混合两种结果
        result = cv2.addWeighted(result1, 0.5, result2, 0.5, 0)
        
        # 后处理
        result = self.post_process(result)
        
        # 生成保存路径
        save_path = day_img_path.replace('cityscapes', 'cityscapes_night')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存结果
        cv2.imwrite(save_path, result)
        
        return i

    def convert_to_night(self, save_path, random_seed=None, num_threads=8):
        """
        使用多线程转换所有日景图像并保存
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 准备处理参数
        process_args = [
            (i, img, path) 
            for i, (img, path) in enumerate(zip(self.day_images, self.day_image_paths))
        ]

        # 创建线程池
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 使用tqdm显示进度
            list(tqdm(
                executor.map(self.process_single_image, process_args),
                total=len(self.day_images),
                desc=f"使用{num_threads}个线程转换图像"
            ))

def main():
    # 创建转换器实例
    converter = DayToNight()
    
    # 设置输入输出路径
    day_path = "data/cityscapes/leftImg8bit/**/**/*.png"
    night_path = "data/nightcity-fine/**/img/*.png"
    save_path = "results"
    
    # 加载图像
    converter.load_images(day_path, night_path)
    
    # 转换所有图像，使用8个线程
    converter.convert_to_night(save_path, random_seed=42, num_threads=32)

if __name__ == "__main__":
    main()