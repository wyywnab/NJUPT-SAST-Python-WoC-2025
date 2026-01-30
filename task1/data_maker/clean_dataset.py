import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm，或者删掉相关代码

# ================= 配置区域 =================
# 数据集路径
HR_FOLDER = 'dataset_output/HR'
LR_FOLDER = 'dataset_output/LR'

# 回收站路径（被清洗的图片会移到这里）
TRASH_HR = 'dataset_output/trash_bin/HR'
TRASH_LR = 'dataset_output/trash_bin/LR'

# 阈值设置 (关键参数)
# 标准差阈值：低于此值的被视为“无内容/纯色”
# 纯色图片标准差为 0。
# 稍微有点噪点的纯色背景通常在 0 ~ 5 之间。
# 建议：先设为 5 或 10 运行一次看看效果。
STD_THRESHOLD = 7.0 
# ===========================================

def clean_images():
    # 1. 准备目录
    os.makedirs(TRASH_HR, exist_ok=True)
    os.makedirs(TRASH_LR, exist_ok=True)

    # 获取所有 HR 图片列表
    if not os.path.exists(HR_FOLDER):
        print(f"找不到文件夹: {HR_FOLDER}")
        return

    image_files = [f for f in os.listdir(HR_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"🔍 开始扫描，共有 {len(image_files)} 张图片...")
    print(f"⚙️ 过滤阈值 (标准差) < {STD_THRESHOLD}")

    moved_count = 0
    
    # 使用 tqdm 显示进度条
    for img_name in tqdm(image_files):
        hr_path = os.path.join(HR_FOLDER, img_name)
        lr_path = os.path.join(LR_FOLDER, img_name)

        # 读取图片 (以灰度模式读取，计算更快且足以判断内容)
        img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ 无法读取文件 (可能是坏图): {img_name}")
            continue

        # === 核心算法：计算像素标准差 ===
        # std 越大，表示图片里像素差异越大（纹理越丰富）
        # std 越小，表示图片越平坦（纯色或渐变）
        img_std = np.std(img)

        # 如果标准差小于阈值，认为是“废片”
        if img_std < STD_THRESHOLD:
            # 移动 HR
            shutil.move(hr_path, os.path.join(TRASH_HR, img_name))
            
            # 移动对应的 LR (保持数据集对齐)
            if os.path.exists(lr_path):
                shutil.move(lr_path, os.path.join(TRASH_LR, img_name))
            
            moved_count += 1
            # 如果你想看具体数值，可以取消下面这行的注释
            # print(f"移出: {img_name} (Score: {img_std:.2f})")

    print("-" * 30)
    print(f"✅ 清洗完成！")
    print(f"🗑️ 共移除了 {moved_count} 张图片。")
    print(f"📂 它们被保存在: {os.path.dirname(TRASH_HR)}")
    print("请去回收站检查一下，确认无误后可手动删除该文件夹。")

if __name__ == '__main__':
    clean_images()
