import os
import yaml
import datetime
import shutil
import logging
import csv
import time
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import SRNet
import random
import numpy as np

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
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))

        assert len(self.hr_images) == len(self.lr_images), "Mismatch in number of HR and LR images"

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_name = self.hr_images[idx]
        lr_name = self.lr_images[idx]

        hr_path = os.path.join(self.hr_dir, hr_name)
        lr_path = os.path.join(self.lr_dir, lr_name)

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    exp_name = f"exp{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    exp_dir = os.path.join("exp", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    shutil.copy("config.yaml", os.path.join(exp_dir, "config.yaml"))

    csv_path = os.path.join(exp_dir, "curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time"])

    torch.cuda.empty_cache()

    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    SCALE_FACTOR = config['scale_factor']
    DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    CRITERION = config['criterion']
    OPTIMIZER = config['optimizer']
    logger.info(f"Using device: {DEVICE}")

    train_hr_dir = "dataset/train/HR"
    train_lr_dir = "dataset/train/LR"
    val_hr_dir = "dataset/val/HR"
    val_lr_dir = "dataset/val/LR"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = SRDataset(train_hr_dir, train_lr_dir, transform=transform)
    val_dataset = SRDataset(val_hr_dir, val_lr_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=config.get("num_workers", 4))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config.get("num_workers", 4))

    model = SRNet(scale_factor=SCALE_FACTOR).to(DEVICE)

    if CRITERION == 'L1Loss':
        criterion = nn.L1Loss()
    elif CRITERION == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {CRITERION}")

    if OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=config.get('weight_decay', 0.01))
    elif OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config.get('weight_decay', 0.01))
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=config.get('momentum', 0.9), weight_decay=config.get('weight_decay', 0.01))
    else:
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

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

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False) # tqdm
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(loop):
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            loop.set_postfix(loss=loss.item()) # tqdm

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(val_loader, desc="Validating", leave=False):
                lr_imgs = lr_imgs.to(DEVICE)
                hr_imgs = hr_imgs.to(DEVICE)

                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, current_lr, epoch_duration])

        if scheduler:
            scheduler.step()

        logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr}, Time: {epoch_duration:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, f"best_model_epoch_{epoch}.pth"))
            logger.info("Saved best model.")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    logger.info(f"Training finished. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()
