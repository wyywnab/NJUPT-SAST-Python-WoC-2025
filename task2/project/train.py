import os
import yaml
import datetime
import shutil
import logging
import csv
import time

from torchvision.datasets import CIFAR10
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import ClassNet
import random
import numpy as np

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

def get_data_loaders(data_root, batch_size, seed, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = CIFAR10(data_root, transform=transform, download=True, train=True)

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_workers)

    return train_loader, val_loader

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
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"])

    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CRITERION = config['criterion']
    OPTIMIZER = config['optimizer']
    logger.info(f"Using device: {DEVICE}")

    train_loader, val_loader = get_data_loaders("./cifar10", BATCH_SIZE, seed, config.get("num_workers", 4))

    model = ClassNet(class_nums=10).to(DEVICE)

    if CRITERION == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
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
        logger.warning(f"Unsupported scheduler: {scheduler_name}. No scheduler will be used.")
        scheduler = None

    early_stopping = EarlyStopping(patience=config.get('patience', 10), min_delta=config.get('min_delta', 0))

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False) # tqdm
        for i, (x, y) in enumerate(loop):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)

            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            train_loss += loss.item()

            # train accuracy
            preds = output.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

            optimizer.step()

            loop.set_postfix(loss=loss.item(), acc=str(train_correct / train_total * 100)[:5] + "%") # tqdm

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)

                output = model.forward(x)
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

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc, current_lr, epoch_duration])

        if scheduler:
            scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Finished. Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr}, Time: {epoch_duration:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, f"best_model_epoch_{epoch + 1}.pth"))
            logger.info(f"Saved best model on epoch {epoch + 1}.")

        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))
    logger.info(f"All done!")

if __name__ == "__main__":
    main()
