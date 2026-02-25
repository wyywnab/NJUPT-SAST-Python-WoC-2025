import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from model_class import ClassNet
from fused_model import AdaptiveSRSystem


class FusedSRDatasetWithPaths(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, str, int]] = []

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

        return lr_img, hr_img, cls_idx, lr_path


def build_gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    window_2d = window_2d / window_2d.sum()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(sr, hr, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    eps = 1e-10
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
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
    return ssim_map.flatten(1).mean(dim=1)


def load_classifier(weights_path: str, num_classes: int, device: torch.device) -> ClassNet:
    model = ClassNet(class_nums=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_sr_system(weights_path: str, num_classes: int, scale_factor: int, device: torch.device) -> AdaptiveSRSystem:
    model = AdaptiveSRSystem(num_classes=num_classes, sr_scale=scale_factor, training_experts_only=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_classifier(classifier: ClassNet, data_root: str, device: torch.device, batch_size: int, num_workers: int):
    test_dir = os.path.join(data_root, "test")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="Classify(Test)",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, dataset.classes


def init_metrics(classes: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {"overall": {"psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0}}
    for cls in classes:
        stats[cls] = {"psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0}
    return stats


def update_metrics(stats: Dict[str, Dict[str, float]], class_name: str, psnr: float, ssim: float):
    stats["overall"]["psnr_sum"] += psnr
    stats["overall"]["ssim_sum"] += ssim
    stats["overall"]["count"] += 1

    stats[class_name]["psnr_sum"] += psnr
    stats[class_name]["ssim_sum"] += ssim
    stats[class_name]["count"] += 1


def finalize_metrics(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    results = {}
    for key, value in stats.items():
        count = value["count"]
        if count == 0:
            results[key] = {"psnr": 0.0, "ssim": 0.0, "count": 0}
        else:
            results[key] = {
                "psnr": value["psnr_sum"] / count,
                "ssim": value["ssim_sum"] / count,
                "count": count,
            }
    return results


def resolve_thumbnail_path(lr_path: str, thumb_root: str, class_name: str) -> str:
    stem = os.path.splitext(os.path.basename(lr_path))[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        base = "_".join(parts[:2])
    else:
        base = stem

    class_dir = os.path.join(thumb_root, class_name)
    for ext in (".jpg", ".png", ".jpeg", ".bmp"):
        candidate = os.path.join(class_dir, base + ext)
        if os.path.exists(candidate):
            return candidate

    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.startswith(base):
                return os.path.join(class_dir, fname)

    raise FileNotFoundError(f"Thumbnail not found for {lr_path}")


def evaluate_sr_oracle(sr_model: AdaptiveSRSystem, dataset: FusedSRDatasetWithPaths, device: torch.device,
                       batch_size: int, num_workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    stats = init_metrics(dataset.classes)

    with torch.no_grad():
        for lr_imgs, hr_imgs, labels, _ in tqdm(
            loader,
            desc="SR(Oracle)",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            labels = labels.to(device)

            sr_out, _ = sr_model(lr_imgs, gt_label=labels)
            sr_out = sr_out.clamp(0.0, 1.0)

            psnr_vals = compute_psnr(sr_out, hr_imgs).cpu().tolist()
            ssim_vals = compute_ssim(sr_out, hr_imgs).cpu().tolist()
            label_vals = labels.cpu().tolist()

            for psnr, ssim, label_idx in zip(psnr_vals, ssim_vals, label_vals):
                class_name = dataset.classes[label_idx]
                update_metrics(stats, class_name, psnr, ssim)

    return finalize_metrics(stats)


def evaluate_sr_mixed(sr_model: AdaptiveSRSystem, classifier: ClassNet, dataset: FusedSRDatasetWithPaths,
                      thumb_root: str, device: torch.device, num_workers: int):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    stats = init_metrics(dataset.classes)

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with torch.no_grad():
        for lr_imgs, hr_imgs, labels, lr_paths in tqdm(
            loader,
            desc="SR(Mixed)",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            class_name = dataset.classes[labels.item()]

            thumb_path = resolve_thumbnail_path(lr_paths[0], thumb_root, class_name)
            thumb_img = Image.open(thumb_path).convert("RGB")
            thumb_tensor = cls_transform(thumb_img).unsqueeze(0).to(device)

            logits = classifier(thumb_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()

            sr_out = sr_model.sr_experts[pred_idx](lr_imgs).clamp(0.0, 1.0)

            psnr_val = compute_psnr(sr_out, hr_imgs).item()
            ssim_val = compute_ssim(sr_out, hr_imgs).item()

            update_metrics(stats, class_name, psnr_val, ssim_val)

    return finalize_metrics(stats)


def evaluate_thumb_classifier(classifier: ClassNet, thumb_root: str, device: torch.device,
                              batch_size: int, num_workers: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root=thumb_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(
            loader,
            desc="Classify(Thumb)",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, dataset.classes


def evaluate_sr_random(sr_model: AdaptiveSRSystem, dataset: FusedSRDatasetWithPaths, device: torch.device,
                       num_workers: int):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    stats = init_metrics(dataset.classes)
    num_experts = len(sr_model.sr_experts)

    with torch.no_grad():
        for lr_imgs, hr_imgs, labels, _ in tqdm(
            loader,
            desc="SR(Random)",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            class_name = dataset.classes[labels.item()]

            rand_idx = torch.randint(0, num_experts, (1,), device=device).item()
            sr_out = sr_model.sr_experts[rand_idx](lr_imgs).clamp(0.0, 1.0)

            psnr_val = compute_psnr(sr_out, hr_imgs).item()
            ssim_val = compute_ssim(sr_out, hr_imgs).item()

            update_metrics(stats, class_name, psnr_val, ssim_val)

    return finalize_metrics(stats)


def evaluate_sr_fixed_expert(sr_model: AdaptiveSRSystem, dataset: FusedSRDatasetWithPaths, device: torch.device,
                             num_workers: int, expert_idx: int):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    stats = init_metrics(dataset.classes)
    num_experts = len(sr_model.sr_experts)

    if expert_idx < 0 or expert_idx >= num_experts:
        raise ValueError(f"Invalid expert index {expert_idx}, total experts: {num_experts}")

    with torch.no_grad():
        for lr_imgs, hr_imgs, labels, _ in tqdm(
            loader,
            desc=f"SR(Fixed-{expert_idx})",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            class_name = dataset.classes[labels.item()]

            sr_out = sr_model.sr_experts[expert_idx](lr_imgs).clamp(0.0, 1.0)

            psnr_val = compute_psnr(sr_out, hr_imgs).item()
            ssim_val = compute_ssim(sr_out, hr_imgs).item()

            update_metrics(stats, class_name, psnr_val, ssim_val)

    return finalize_metrics(stats)


def print_metrics(title: str, results: Dict[str, Dict[str, float]]):
    print(f"\n{title}")
    overall = results["overall"]
    print(f"Overall: PSNR={overall['psnr']:.4f}, SSIM={overall['ssim']:.4f} (N={overall['count']})")
    for cls_name, metrics in results.items():
        if cls_name == "overall":
            continue
        print(f"  {cls_name}: PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f} (N={metrics['count']})")
    print("\n")


def print_delta(title: str, a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]):
    print(f"\n{title}")
    for key in a.keys():
        if key not in b:
            continue
        delta_psnr = a[key]["psnr"] - b[key]["psnr"]
        delta_ssim = a[key]["ssim"] - b[key]["ssim"]
        label = "Overall" if key == "overall" else f"  {key}"
        print(f"{label}: PSNR={delta_psnr:+.4f}, SSIM={delta_ssim:+.4f}")


def write_results_csv(
    output_path: str,
    cls_acc: Optional[float] = None,
    thumb_acc: Optional[float] = None,
    cls_names: Optional[List[str]] = None,
    thumb_names: Optional[List[str]] = None,
    oracle_results: Optional[Dict[str, Dict[str, float]]] = None,
    mixed_results: Optional[Dict[str, Dict[str, float]]] = None,
    random_results: Optional[Dict[str, Dict[str, float]]] = None,
    fixed_results: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None,
):
    rows = []

    if cls_acc is not None:
        rows.append({
            "section": "classification",
            "class": "overall",
            "psnr": "",
            "ssim": "",
            "count": "",
            "acc": f"{cls_acc:.6f}",
            "notes": "test",
        })
    if cls_names is not None:
        rows.append({
            "section": "classification",
            "class": "names",
            "psnr": "",
            "ssim": "",
            "count": "",
            "acc": "",
            "notes": ",".join(cls_names),
        })
    if thumb_acc is not None:
        rows.append({
            "section": "thumbnail",
            "class": "overall",
            "psnr": "",
            "ssim": "",
            "count": "",
            "acc": f"{thumb_acc:.6f}",
            "notes": "thumb",
        })
    if thumb_names is not None:
        rows.append({
            "section": "thumbnail",
            "class": "names",
            "psnr": "",
            "ssim": "",
            "count": "",
            "acc": "",
            "notes": ",".join(thumb_names),
        })

    def append_metrics(section: str, results: Dict[str, Dict[str, float]]):
        for key, value in results.items():
            rows.append({
                "section": section,
                "class": key,
                "psnr": f"{value['psnr']:.6f}",
                "ssim": f"{value['ssim']:.6f}",
                "count": f"{value['count']}",
                "acc": "",
                "notes": "",
            })

    if oracle_results is not None:
        append_metrics("sr_oracle", oracle_results)
    if mixed_results is not None:
        append_metrics("sr_mixed", mixed_results)
    if random_results is not None:
        append_metrics("sr_random", random_results)
    if fixed_results is not None:
        for expert_idx, results in fixed_results.items():
            append_metrics(f"sr_fixed_{expert_idx}", results)

    def append_delta(section: str, a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]):
        for key in a.keys():
            if key not in b:
                continue
            rows.append({
                "section": section,
                "class": key,
                "psnr": f"{(a[key]['psnr'] - b[key]['psnr']):+.6f}",
                "ssim": f"{(a[key]['ssim'] - b[key]['ssim']):+.6f}",
                "count": "",
                "acc": "",
                "notes": "",
            })

    if mixed_results is not None and oracle_results is not None:
        append_delta("delta_mixed_oracle", mixed_results, oracle_results)
    if random_results is not None and oracle_results is not None:
        append_delta("delta_random_oracle", random_results, oracle_results)
    if random_results is not None and mixed_results is not None:
        append_delta("delta_random_mixed", random_results, mixed_results)
    if fixed_results is not None and oracle_results is not None:
        for expert_idx, results in fixed_results.items():
            append_delta(f"delta_fixed_{expert_idx}_oracle", results, oracle_results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "class", "psnr", "ssim", "count", "acc", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Test classification, SR, and mixed performance.")
    parser.add_argument("--type", choices=["class", "sr", "fused"], default="fused",
                        help="Evaluation type: class, sr, or fused (default)")
    parser.add_argument("--class-data", default="./dataset_class", help="Root of classification dataset")
    parser.add_argument("--sr-data", default="./dataset_sr", help="Root of SR dataset")
    parser.add_argument("--class-weights", required=True, help="Path to classifier weights")
    parser.add_argument("--sr-weights", required=True, help="Path to SR (fused) weights")
    parser.add_argument("--scale", type=int, default=2, help="SR scale factor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-batch", type=int, default=32)
    parser.add_argument("--sr-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expert", type=int, choices=[0, 1, 2], default=None,
                        help="Only evaluate one fixed expert (0/1/2) for SR and skip other SR evaluations")
    parser.add_argument("--seed", type=int, default=608, help="Random seed for reproducibility")
    parser.add_argument("--out-csv", default="eval_results.csv", help="Path to save csv results")
    args = parser.parse_args()

    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_class = args.type in ("class", "fused")
    run_sr = args.type in ("sr", "fused")
    expert_only = args.expert is not None

    if run_class and not args.class_weights:
        raise ValueError("--class-weights is required when --type is class or fused")
    if run_sr and not args.sr_weights:
        raise ValueError("--sr-weights is required when --type is sr or fused")
    if run_sr and not expert_only and not args.class_weights:
        raise ValueError("--class-weights is required for SR mixed evaluation when --expert is not set")

    cls_acc: Optional[float] = None
    cls_names: Optional[List[str]] = None
    thumb_acc: Optional[float] = None
    thumb_names: Optional[List[str]] = None
    oracle_results: Optional[Dict[str, Dict[str, float]]] = None
    mixed_results: Optional[Dict[str, Dict[str, float]]] = None
    random_results: Optional[Dict[str, Dict[str, float]]] = None
    fixed_results: Dict[int, Dict[str, Dict[str, float]]] = {}

    classifier: Optional[ClassNet] = None
    num_classes: Optional[int] = None

    if run_class:
        class_test_dir = os.path.join(args.class_data, "test")
        if not os.path.isdir(class_test_dir):
            raise FileNotFoundError(f"Classification test dir not found: {class_test_dir}")

        class_dataset = datasets.ImageFolder(
            root=class_test_dir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
        num_classes = len(class_dataset.classes)
        classifier = load_classifier(args.class_weights, num_classes, device)

        cls_acc, cls_names = evaluate_classifier(
            classifier, args.class_data, device, args.class_batch, args.num_workers
        )
        print(f"Classification Acc: {cls_acc * 100:.2f}%")
        print(f"Classes: {cls_names}")

        if args.type == "fused":
            sr_thumb_dir = os.path.join(args.sr_data, "thumb")
            if not os.path.isdir(sr_thumb_dir):
                raise FileNotFoundError(f"SR thumb dir not found: {sr_thumb_dir}")
            thumb_acc, thumb_names = evaluate_thumb_classifier(
                classifier, sr_thumb_dir, device, args.class_batch, args.num_workers
            )
            print(f"Thumbnail Acc: {thumb_acc * 100:.2f}%")
            print(f"Thumb Classes: {thumb_names}")

    if run_sr:
        sr_test_dir = os.path.join(args.sr_data, "test")
        if not os.path.isdir(sr_test_dir):
            raise FileNotFoundError(f"SR test dir not found: {sr_test_dir}")

        sr_dataset = FusedSRDatasetWithPaths(
            sr_test_dir,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )

        if num_classes is None:
            num_classes = len(sr_dataset.classes)
        elif num_classes != len(sr_dataset.classes):
            print("Warning: class count mismatch between classification and SR datasets.")

        sr_model = load_sr_system(args.sr_weights, num_classes, args.scale, device)

        if expert_only:
            if args.expert >= len(sr_model.sr_experts):
                raise ValueError(
                    f"--expert {args.expert} out of range for loaded model with {len(sr_model.sr_experts)} experts"
                )
            fixed_results[args.expert] = evaluate_sr_fixed_expert(
                sr_model, sr_dataset, device, args.num_workers, args.expert
            )
            print_metrics(f"SR (Fixed Expert {args.expert})", fixed_results[args.expert])
        else:
            if classifier is None:
                classifier = load_classifier(args.class_weights, num_classes, device)

            sr_thumb_dir = os.path.join(args.sr_data, "thumb")
            if not os.path.isdir(sr_thumb_dir):
                raise FileNotFoundError(f"SR thumb dir not found: {sr_thumb_dir}")

            oracle_results = evaluate_sr_oracle(
                sr_model, sr_dataset, device, args.sr_batch, args.num_workers
            )
            print_metrics("SR (Oracle by GT class)", oracle_results)

            mixed_results = evaluate_sr_mixed(
                sr_model, classifier, sr_dataset, sr_thumb_dir, device, args.num_workers
            )
            print_metrics("SR (Mixed: Thumbnail -> Class -> Expert)", mixed_results)

            random_results = evaluate_sr_random(
                sr_model, sr_dataset, device, args.num_workers
            )
            print_metrics("SR (Random Expert)", random_results)

            max_fixed = min(3, len(sr_model.sr_experts))
            if max_fixed < 3:
                print(f"Warning: only {len(sr_model.sr_experts)} experts found, evaluating 0..{max_fixed - 1}.")
            for expert_idx in range(max_fixed):
                fixed_results[expert_idx] = evaluate_sr_fixed_expert(
                    sr_model, sr_dataset, device, args.num_workers, expert_idx
                )
                print_metrics(f"SR (Fixed Expert {expert_idx})", fixed_results[expert_idx])

            print_delta("Mixed vs Oracle (Mixed - Oracle)", mixed_results, oracle_results)
            print_delta("Random vs Oracle (Random - Oracle)", random_results, oracle_results)
            print_delta("Random vs Mixed (Random - Mixed)", random_results, mixed_results)
            for expert_idx, results in fixed_results.items():
                print_delta(
                    f"Fixed-{expert_idx} vs Oracle (Fixed-{expert_idx} - Oracle)",
                    results,
                    oracle_results,
                )

    write_results_csv(
        args.out_csv,
        cls_acc=cls_acc,
        thumb_acc=thumb_acc,
        cls_names=cls_names,
        thumb_names=thumb_names,
        oracle_results=oracle_results,
        mixed_results=mixed_results,
        random_results=random_results,
        fixed_results=fixed_results,
    )
    print(f"Results saved to CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
