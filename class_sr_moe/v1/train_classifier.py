import os
import yaml
import datetime
import shutil
import logging
import csv
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model_class import ClassNet


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_data_loaders(data_root, batch_size, num_workers):
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"数据集目录结构不完整 (需包含 train 和 val): {data_root}")

    # 使用ImageFolder加载
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    print(f"Class mapping detected: {train_dataset.class_to_idx}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, val_loader, len(train_dataset.classes)

def main():
    # 配置读取
    if not os.path.exists("config_class.yaml"):
        raise FileNotFoundError("config_class.yaml not found!")

    with open("config_class.yaml", "r") as f:
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
    exp_name = f"exp_class_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
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
    if os.path.exists("config_class.yaml"):
        shutil.copy("config_class.yaml", os.path.join(exp_dir, "config_class.yaml"))

    # 初始化csv文件
    csv_path = os.path.join(exp_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    # 必要参数
    BATCH_SIZE = config.get('batch_size', 16)
    LEARNING_RATE = config.get('learning_rate', 1e-3)
    NUM_EPOCHS = config.get('num_epochs', 50)
    CRITERION = config['criterion']
    OPTIMIZER = config['optimizer']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # 加载数据
    data_root = "./dataset_class"
    train_loader, val_loader, num_classes = get_data_loaders(data_root, BATCH_SIZE, config.get("num_workers", 4))
    logger.info(f"Data loaded. Num classes: {num_classes}")

    # 初始化模型
    model = ClassNet(class_nums=num_classes).to(DEVICE)

    # 损失函数
    if CRITERION == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {CRITERION}")

    # 优化器
    weight_decay = config.get('weight_decay', 0.01)

    if OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    elif OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=config.get('momentum', 0.9), weight_decay=weight_decay)
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
        scheduler = None

    early_stopping = EarlyStopping(patience=config.get('patience', 10), min_delta=config.get('min_delta', 0))

    best_val_acc = 0.0

    # 训练主循环
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # tqdm进度条
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train", leave=False)

        # batch
        for i, (x, y) in enumerate(loop):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()

            # 计算准确率
            preds = output.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

            optimizer.step()
            loop.set_postfix(loss=loss.item(), acc=f"{train_correct / train_total * 100:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)

                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

                preds = output.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 写入csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc, current_lr, epoch_duration])

        if scheduler:
            scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.1e}"
        )

        # 保存最佳模型(acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(exp_dir, f"best_class_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model (Acc: {val_acc:.4f}) on epoch {epoch + 1}")

        # 早停
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(exp_dir, "final_class_model.pth"))
    logger.info(f"All done! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
