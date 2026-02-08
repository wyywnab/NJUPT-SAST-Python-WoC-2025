import os
from PIL import Image
from torch.utils.data import Dataset


class FusedSRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (lr_path, hr_path, class_idx)

        # 扫描类别文件夹
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"🔍 Found classes in {root_dir}: {self.class_to_idx}")

        # 遍历每个类别文件夹
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            hr_dir = os.path.join(root_dir, cls_name, "HR")
            lr_dir = os.path.join(root_dir, cls_name, "LR")

            # 检查文件夹是否存在
            if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
                print(f"⚠️ Warning: Missing HR or LR folder in {cls_name}, skipping.")
                continue

            filenames = sorted(os.listdir(hr_dir))
            for fname in filenames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    hr_path = os.path.join(hr_dir, fname)
                    lr_path = os.path.join(lr_dir, fname)

                    # 只有对应lr图像也存在才加
                    if os.path.exists(lr_path):
                        self.samples.append((lr_path, hr_path, cls_idx))
                    else:
                        print(f"⚠️ Warning: LR image not found for {fname}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_path, hr_path, cls_idx = self.samples[idx]

        # 读取图片
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        # 应用变换
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img, cls_idx
