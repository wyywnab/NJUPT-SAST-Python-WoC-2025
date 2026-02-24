import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# 引入你的模型定义
from fused_model import AdaptiveSRSystem

# ==========================================
# 1. 必要的辅助函数 (PSNR, SSIM, Dataset)
# ==========================================

def build_gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    eps = 1e-10
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean().item()

def compute_ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    channels = sr.shape[1]
    window = build_gaussian_window(window_size, sigma, channels, sr.device, sr.dtype)
    padding = window_size // 2

    mu1 = F.conv2d(sr, window, padding=padding, groups=channels)
    mu2 = F.conv2d(hr, window, padding=padding, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=padding, groups=channels) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.flatten(1).mean(dim=1).mean().item()

class FusedSRDatasetWithPaths(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # 自动读取子文件夹作为类别
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            hr_dir = os.path.join(root_dir, cls_name, "HR")
            lr_dir = os.path.join(root_dir, cls_name, "LR")
            if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
                continue
            
            filenames = sorted(os.listdir(hr_dir))
            for fname in filenames:
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    hr_path = os.path.join(hr_dir, fname)
                    lr_path = os.path.join(lr_dir, fname)
                    if os.path.exists(lr_path):
                        self.samples.append((lr_path, hr_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_path, hr_path, cls_idx = self.samples[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img, cls_idx

# ==========================================
# 2. 核心评估逻辑
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Simplified Adaptive SR Evaluation")
    parser.add_argument("--data_root", type=str, default="task1\project\dataset", help="Path to dataset root (e.g., dataset/test)")
    parser.add_argument("--checkpoint", type=str, default="class_sr_moe\\v2\\weights\\fused\\best_fused_model_epoch_52.pth", help="Path to .pth model file")
    parser.add_argument("--scale", type=int, default=2, help="SR Scale factor")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(), # 自动归一化到 [0, 1]
    ])
    
    dataset = FusedSRDatasetWithPaths(root_dir=args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Dataset loaded: {len(dataset)} images from {len(dataset.classes)} classes: {dataset.classes}")

    # 2. 加载模型
    # training_experts_only=True 表示不加载分类器部分，只加载超分专家
    model = AdaptiveSRSystem(num_classes=len(dataset.classes), sr_scale=args.scale, training_experts_only=True)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        # strict=False 允许我们忽略掉checkpoint里可能存在的分类器权重(如果有的话)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    model.to(device)
    model.eval()

    for expert in range(3):
        eval_single(dataloader, device, model, dataset,  expert)

def eval_single(dataloader, device, model, dataset, expert):

    # 3. 开始评估
    metrics = {cls: {"psnr": 0.0, "ssim": 0.0, "count": 0} for cls in dataset.classes}
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0

    print(f"\nStarting Evaluation (Using Expert {expert})...")
    
    with torch.no_grad():
        for lr, hr, label in tqdm(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            label = label.to(device)

            # 关键点：传入 gt_label，模型会自动选择对应的 Expert 进行推理
            # 这样就跳过了分类器，直接评估超分性能
            forced_label = torch.full_like(label, fill_value=expert) 
            sr, _ = model(lr, gt_label=forced_label)
            
            # 确保输出在 [0, 1] 范围内
            sr = sr.clamp(0.0, 1.0)

            # 计算指标
            psnr_val = compute_psnr(sr, hr)
            ssim_val = compute_ssim(sr, hr)

            # 记录
            cls_name = dataset.classes[label.item()]
            metrics[cls_name]["psnr"] += psnr_val
            metrics[cls_name]["ssim"] += ssim_val
            metrics[cls_name]["count"] += 1

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_count += 1

    # 4. 输出结果
    print("\n" + "="*40)
    print(f" Evaluation Results (Expert {expert})")
    print("="*40)
    
    print(f"{'Class':<15} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'Count':<5}")
    print("-" * 48)

    for cls in dataset.classes:
        data = metrics[cls]
        if data["count"] > 0:
            avg_psnr = data["psnr"] / data["count"]
            avg_ssim = data["ssim"] / data["count"]
            print(f"{cls:<15} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | {data['count']:<5}")

    print("-" * 48)
    if total_count > 0:
        avg_total_psnr = total_psnr / total_count
        avg_total_ssim = total_ssim / total_count
        print(f"{'OVERALL':<15} | {avg_total_psnr:<10.4f} | {avg_total_ssim:<10.4f} | {total_count:<5}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
