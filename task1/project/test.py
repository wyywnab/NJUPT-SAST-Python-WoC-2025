import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from model import SRNet

def process_image(model, img_path, save_path, device, to_tensor, to_pil):
    try:
        lr_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return

    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    sr_tensor = model(lr_tensor)

    sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)
    sr_img = to_pil(sr_tensor)

    sr_img.save(save_path)
    print(f"Image saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--model', type=str, default='exp/exp260128_160539/best_model_epoch_93.pth')
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model):
        print("Model Not Found.")

    print(f"Loading model from {args.model}...")
    model = SRNet(scale_factor=args.scale).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if os.path.isfile(args.input):
        filename = os.path.basename(args.input)
        save_path = os.path.join(args.output, filename)
        with torch.no_grad():
            process_image(model, args.input, save_path, device, to_tensor, to_pil)

    elif os.path.isdir(args.input):
        image_files = sorted(os.listdir(args.input))
        print(f"Processing {len(image_files)} files in {args.input}")

        with torch.no_grad():
            for img_name in image_files:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                img_path = os.path.join(args.input, img_name)
                save_path = os.path.join(args.output, img_name)
                process_image(model, img_path, save_path, device, to_tensor, to_pil)
    else:
        print(f"Error: '{args.input}' is not a file or directory.")

if __name__ == "__main__":
    main()

