import os
import yaml
import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from model import ClassNet


def select_checkpoint_from_exp(exp_dir, preferred_name=None):
    if not os.path.isdir(exp_dir):
        return None

    pth_files = []
    for entry in os.listdir(exp_dir):
        path = os.path.join(exp_dir, entry)
        if os.path.isfile(path) and entry.endswith('.pth'):
            pth_files.append(path)
    if preferred_name:
        pref_path = os.path.join(exp_dir, preferred_name)
        if os.path.isfile(pref_path):
            return pref_path
    if not pth_files:
        return None

    bests = [p for p in pth_files if os.path.basename(p).startswith('best_model')]
    if bests:
        return max(bests, key=lambda p: os.path.getmtime(p))
    finals = [p for p in pth_files if os.path.basename(p) == 'final_model.pth']
    if finals:
        return max(finals, key=lambda p: os.path.getmtime(p))
    return max(pth_files, key=lambda p: os.path.getmtime(p))


def get_test_loader(data_root, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = CIFAR10(data_root, transform=transform, download=False, train=False)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return loader


def load_config_from_exp(exp_dir):
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)

    if os.path.isfile('config.yaml'):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    return {}

def evaluate(exp_dir, model_file=None, device=None, batch_size=None, num_workers=None, data_root='./cifar10'):
    logger = logging.getLogger()
    cfg = load_config_from_exp(exp_dir) or {}

    data_root = data_root or './cifar10'
    batch_size = batch_size or cfg.get('batch_size', 128)
    num_workers = num_workers if num_workers is not None else cfg.get('num_workers', 4)

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    ckpt_path = select_checkpoint_from_exp(exp_dir, preferred_name=model_file)
    if ckpt_path is None:
        raise FileNotFoundError(f'No .pth checkpoint found in {exp_dir}')
    logger.info(f'Using checkpoint: {ckpt_path}')

    model = ClassNet(class_nums=10)
    model.to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = get_test_loader(data_root, batch_size, num_workers)

    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    duration = time.time() - start
    acc = correct / total if total > 0 else 0.0

    results_path = os.path.join(exp_dir, 'eval_results.txt')
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    report = f"{timestamp} | model: {os.path.basename(ckpt_path)} | test_acc: {acc:.4f} | correct: {correct} | total: {total} | time: {duration:.2f}s\n"
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(report)

    logger.info(report.strip())
    print(report.strip())
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default="exp/exp260131_105143")
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if not os.path.isdir(args.exp_dir):
        raise FileNotFoundError(f'Experiment directory not found: {args.exp_dir}')

    evaluate(args.exp_dir, model_file=args.model_file, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root)


if __name__ == '__main__':
    main()
