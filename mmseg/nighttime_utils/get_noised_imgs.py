import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

from concurrent.futures import ThreadPoolExecutor
import threading
from dark_noising import AddNoisyImg

class Noiser:
    def __init__(self):
        self.night_features = None
        self.noiser = AddNoisyImg(model='PGRU',
                                camera='CanonEOS5D4',
                                cfa='bayer',
                                use_255=True,
                                pre_adjust_brightness=False,
                                mode='addnoise',
                                dark_ratio=(1.0, 1.0),
                                noise_ratio=(10, 100))
        
    def load_images(self, day_path):
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
                    
        print(f"已加载 {len(self.day_images)} 张日景图像")
        
    
    
    def process_single_image(self, args):
        """
        处理单张图像
        """
        i, day_img, day_img_path = args
        result = self.noiser.add_noise_255(day_img)
        
        # 生成保存路径
        save_path = day_img_path.replace('cityscapes', 'cityscapes_noised')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存结果
        cv2.imwrite(save_path, result)
        
        return i

    def convert_to_noised(self, save_path, random_seed=None, num_threads=8):
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
    converter = Noiser()
    
    # 设置输入输出路径
    day_path = "data/cityscapes/leftImg8bit/**/**/*.png"
    save_path = "results"
    
    # 加载图像
    converter.load_images(day_path)
    
    # 转换所有图像，使用8个线程
    converter.convert_to_noised(save_path, random_seed=42, num_threads=32)

if __name__ == "__main__":
    main()