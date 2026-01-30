import os
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import SRNet

def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scale_factor = 2
    model_path = "exp/exp260128_160539/best_model_epoch_93.pth"
    if not os.path.exists(model_path):
        model_path = "final_model.pth"

    lr_dir = "dataset/test/LR"
    hr_dir = "dataset/test/HR"

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = SRNet(scale_factor=scale_factor).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    if not os.path.exists(lr_dir) or not os.path.exists(hr_dir):
        print(f"Error: One of the directories '{lr_dir}' or '{hr_dir}' not found.")
        return

    image_files = sorted(os.listdir(lr_dir))
    print(f"Found {len(image_files)} images in {lr_dir}")

    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for img_name in image_files:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            lr_path = os.path.join(lr_dir, img_name)
            hr_path = os.path.join(hr_dir, img_name)

            if not os.path.exists(hr_path):
                continue

            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = Image.open(hr_path).convert('RGB')

            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

            sr_tensor = model(lr_tensor)

            sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
            sr_img = to_pil(sr_tensor)

            resample_method = Image.Resampling.BICUBIC

            if sr_img.size != hr_img.size:
                hr_img = hr_img.resize(sr_img.size, resample_method)

            psnr = calculate_psnr(sr_img, hr_img)
            total_psnr += psnr
            count += 1

            if count % 10 == 0:
                 print(f"evaluating: {count}/{len(image_files)} | Current avg PSNR: {total_psnr/count:.2f} dB")

    print("-" * 50)
    if count > 0:
        avg_psnr = total_psnr / count
        print(f"Average PSNR of {count} images: {avg_psnr:.2f} dB")
    else:
        print("No matching HR/LR images found for evaluation.")

if __name__ == "__main__":
    main()

