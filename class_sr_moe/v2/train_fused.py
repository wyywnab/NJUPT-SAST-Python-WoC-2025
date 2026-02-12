import os
import yaml
import datetime
import shutil
import logging
import csv
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from fused_model import AdaptiveSRSystem
from data_loader import FusedSRDataset


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta: # loss没有明显下降
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_data_loaders(data_root, batch_size, num_workers):
    transform = transforms.ToTensor() # 只转换为张量，不归一化

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_set = FusedSRDataset(train_dir, transform=transform)
    val_set = FusedSRDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, len(train_set.classes)

def main():
    # 配置读取
    if not os.path.exists("config_fused.yaml"):
        raise FileNotFoundError("config_fused.yaml not found!")

    with open("config_fused.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 设置种子
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 实验目录
    exp_name = f"exp_fused_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    exp_dir = os.path.join("exp", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 日志
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, "train.log"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # 备份配置
    shutil.copy("config_fused.yaml", os.path.join(exp_dir, "config_fused.yaml"))

    # 初始化csv文件
    csv_path = os.path.join(exp_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time"])

    # 必要参数
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    CRITERION = config['criterion']
    OPTIMIZER = config['optimizer']
    SCALE_FACTOR = config['scale_factor']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {DEVICE}")

    # 加载数据
    data_root = "./dataset_sr"
    train_loader, val_loader, num_classes = get_data_loaders(data_root, BATCH_SIZE, config.get("num_workers", 4))
    logger.info(f"Data loaded. Num classes: {num_classes}")

    # 初始化模型
    model = AdaptiveSRSystem(
        num_classes=num_classes,
        sr_scale=SCALE_FACTOR,
        pretrained_classifier_path=None, # 不加载分类器
        training_experts_only=True  # 只训练专家
    ).to(DEVICE)

    # 损失函数
    if CRITERION == 'L1Loss':
        criterion = nn.L1Loss()
    elif CRITERION == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {CRITERION}")

    # 只优化requires_grad=True的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 优化器
    weight_decay = config.get('weight_decay', 0.01)
    if OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=weight_decay)
    elif OPTIMIZER == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=weight_decay)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=LEARNING_RATE, momentum=config.get('momentum', 0.9), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

    # 调度器
    scheduler_name = config.get("scheduler", "StepLR")
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 20),
            gamma=config.get("gamma", 0.5)
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", 50)),
            eta_min=float(config.get("eta_min", 1e-6))
        )
    else:
        logger.warning(f"Unsupported scheduler: {scheduler_name}. No scheduler will be used.")
        scheduler = None

    early_stopping = EarlyStopping(patience=config.get('patience', 10), min_delta=config.get('min_delta', 0))

    best_val_loss = float('inf')

    # 训练主循环
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # tqdm进度条
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train", leave=False)

        # batch
        for batch_idx, (lr_imgs, hr_imgs, labels) in enumerate(loop):
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # 传入标签训练对应专家
            sr_outputs, _ = model(lr_imgs, gt_label=labels)

            loss = criterion(sr_outputs, hr_imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                lr_imgs = lr_imgs.to(DEVICE)
                hr_imgs = hr_imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 验证sr效果
                sr_outputs, _ = model(lr_imgs, gt_label=labels)

                loss = criterion(sr_outputs, hr_imgs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 写入csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_duration])

        if scheduler:
            scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Finished. Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}, Time: {epoch_duration:.2f}s")

        # 保存最佳模型(loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, f"best_fused_model_epoch_{epoch + 1}.pth"))
            logger.info(f"Saved best model on epoch {epoch + 1}.")

        # 早停
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_fused_model.pth"))
    logger.info(f"All done! Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
