import os
import shutil
import random
import argparse
from tqdm import tqdm

# ================= 配置区域 =================
# 原始数据路径
SOURCE_ROOT = 'dataset_output'
SRC_HR = os.path.join(SOURCE_ROOT, 'HR')
SRC_LR = os.path.join(SOURCE_ROOT, 'LR')

# 划分比例 (和必须为 1.0)
RATIO_TRAIN = 0.7  # 80% 训练
RATIO_VAL   = 0.15  # 10% 验证
RATIO_TEST  = 0.15  # 10% 测试

# 随机种子 (保证每次运行打乱的结果一致，方便复现)
RANDOM_SEED = 42
# ===========================================

def split_dataset():
    # 1. 检查源文件夹
    if not os.path.exists(SRC_HR) or not os.path.exists(SRC_LR):
        print(f"❌ 错误：找不到源文件夹 {SRC_HR} 或 {SRC_LR}")
        return

    # 2. 获取所有图片文件名
    # 我们以 HR 文件夹为基准
    all_files = [f for f in os.listdir(SRC_HR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_files = len(all_files)

    if total_files == 0:
        print("❌ 文件夹为空，没有图片可划分。")
        return

    print(f"📦 扫描到 {total_files} 张图片，准备划分...")

    # 3. 打乱顺序
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # 4. 计算切分点
    train_end = int(total_files * RATIO_TRAIN)
    val_end = int(total_files * (RATIO_TRAIN + RATIO_VAL))

    # 5. 分配列表
    splits = {
        'train': all_files[:train_end],
        'val':   all_files[train_end:val_end],
        'test':  all_files[val_end:]
    }

    print(f"📊 划分详情: Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 6. 执行移动操作
    for split_name, file_list in splits.items():
        # 创建目标文件夹，例如: dataset_output/train/HR
        target_hr_dir = os.path.join(SOURCE_ROOT, split_name, 'HR')
        target_lr_dir = os.path.join(SOURCE_ROOT, split_name, 'LR')
        
        os.makedirs(target_hr_dir, exist_ok=True)
        os.makedirs(target_lr_dir, exist_ok=True)

        print(f"🚀 正在处理 {split_name} 集...")
        
        for filename in tqdm(file_list):
            # 源路径
            src_hr_path = os.path.join(SRC_HR, filename)
            src_lr_path = os.path.join(SRC_LR, filename)

            # 目标路径
            dst_hr_path = os.path.join(target_hr_dir, filename)
            dst_lr_path = os.path.join(target_lr_dir, filename)

            try:
                # 移动 HR
                shutil.copy(src_hr_path, dst_hr_path)
                
                # 移动 LR (如果存在)
                if os.path.exists(src_lr_path):
                    shutil.copy(src_lr_path, dst_lr_path)
                else:
                    print(f"⚠️ 警告: 找不到对应的 LR 图片 -> {filename}")
            
            except Exception as e:
                print(f"❌ 移动失败 {filename}: {e}")

    # # 7. 清理空文件夹
    # try:
    #     os.rmdir(SRC_HR)
    #     os.rmdir(SRC_LR)
    #     print("🧹 已删除原始空文件夹。")
    # except:
    #     pass # 如果文件夹非空则保留

    print("\n✅ 数据集划分完成！")

if __name__ == '__main__':
    split_dataset()
