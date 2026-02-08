import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fused_model import AdaptiveSRSystem


def inference_whole_image(
    image_path,
    model,
    device,
    patch_size=112,  # 输入网络的切片大小
    overlap=16,  # 边缘重叠大小(将被切除的边缘宽度)
    scale=2,
    output_path=None,
):

    # 读取与处理
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # 全局分类（缩放到224*224）
    classifier_input = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        class_logits = model.classifier(classifier_input)
        scene_idx = torch.argmax(class_logits, dim=1).item()

    print(f"🔍 Scene Analysis: Detected Class {scene_idx}")
    selected_expert = model.sr_experts[scene_idx]

    # ===== 无缝分块超分 =====
    # 计算步长(stride)，即每次移动的距离
    stride = patch_size - (2 * overlap)

    # 对原图进行反射填充
    # 边缘的切片也能获得上下文，且不会产生黑边
    img_padded = F.pad(img_tensor, (overlap, overlap, overlap, overlap), mode='reflect')
    pad_h, pad_w = img_padded.shape[2], img_padded.shape[3]

    # 初始化输出画布
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((1, 3, out_h, out_w), device=device)

    # 双重循环遍历(在填充后的图像上滑动)
    # y, x 是在“原始图像”坐标系下的起始点
    for y in range(0, h, stride):
        for x in range(0, w, stride):

            # --- 提取输入切片 ---
            # 覆盖原图的 [y : y+stride] 区域
            # 在 Padded 图中，该区域对应的坐标是 [y+overlap : y+stride+overlap]
            # 为了获得上下文，我们向外扩展 overlap，所以取 [y : y+stride+2*overlap]
            # 即：[y : y+patch_size]

            in_y_start = y
            in_x_start = x
            in_y_end = in_y_start + patch_size
            in_x_end = in_x_start + patch_size

            # 边界处理：如果超出了填充后的图像边界，就只取最后能取到的部分
            if in_y_end > pad_h:
                in_y_start = pad_h - patch_size
                in_y_end = pad_h
            if in_x_end > pad_w:
                in_x_start = pad_w - patch_size
                in_x_end = pad_w

            in_patch = img_padded[:, :, in_y_start:in_y_end, in_x_start:in_x_end]

            # --- 专家推理 ---
            with torch.no_grad():
                sr_patch = selected_expert(in_patch)

            # --- 裁剪与拼接 ---
            # sr_patch 的大小是 (patch_size * scale)
            # 只保留中间的有效区域，切除四周的 overlap * scale

            # 计算输出切片中“有效区域”的起止点
            out_crop_start = overlap * scale
            out_crop_end = (patch_size - overlap) * scale

            # 1. 确定 sr_patch 对应的原图输出坐标
            # 因为输入是 img_padded[y_start...]，它对应的原图坐标是 (y_start - overlap)
            # 所以输出对应的原图坐标是 (y_start - overlap) * scale
            abs_y_start = (in_y_start - overlap) * scale
            abs_x_start = (in_x_start - overlap) * scale

            # 2. 裁剪掉边缘 (去除 artifacts)
            valid_sr = sr_patch[:, :, out_crop_start:out_crop_end, out_crop_start:out_crop_end]

            # 3. 计算粘贴到大图的位置
            # 有效区域在原图中的起始位置
            paste_y = abs_y_start + out_crop_start
            paste_x = abs_x_start + out_crop_start

            paste_h, paste_w = valid_sr.shape[2], valid_sr.shape[3]

            # 4. 粘贴 (注意边界检查，防止溢出)
            # 只有当 paste_y >= 0 时才粘贴
            y1 = max(0, paste_y)
            x1 = max(0, paste_x)
            y2 = min(out_h, paste_y + paste_h)
            x2 = min(out_w, paste_x + paste_w)

            # 对应的 valid_sr 内部切片
            vy1 = y1 - paste_y
            vx1 = x1 - paste_x
            vy2 = vy1 + (y2 - y1)
            vx2 = vx1 + (x2 - x1)

            output[:, :, y1:y2, x1:x2] = valid_sr[:, :, vy1:vy2, vx1:vx2]

    # 保存结果
    to_pil = transforms.ToPILImage()
    result_img = to_pil(output.squeeze(0).cpu().clamp(0, 1))
    save_name = output_path or f"result_class_{scene_idx}.png"
    result_img.save(save_name)
    print(f"Done! Saved seamless result to {save_name} using Expert {scene_idx}")
    return save_name


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_system_for_inference(sr_weights_path, classifier_weights_path, num_classes, sr_scale=2):
    # 初始化系统
    model = AdaptiveSRSystem(
        num_classes=num_classes,
        sr_scale=sr_scale,
        training_experts_only=False
    )

    # 加载sr权重
    print(f"Loading SR weights from {sr_weights_path}...")
    sr_state = torch.load(sr_weights_path, map_location='cpu')
    model.load_state_dict(sr_state, strict=False)

    # 加载分类器权重
    print(f"Loading Classifier weights from {classifier_weights_path}...")
    cls_state = torch.load(classifier_weights_path, map_location='cpu')
    model.classifier.load_state_dict(cls_state)

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seamless tiled SR inference.")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--input-dir", default="test_input", help="Folder containing input images")
    parser.add_argument("--output-dir", default="test_output", help="Folder to save outputs for batch mode")
    parser.add_argument("--sr-weights", required=True, help="Path to SR (fused) weights")
    parser.add_argument("--class-weights", required=True, help="Path to classifier weights")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--scale", type=int, default=2, help="SR scale factor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, default=112, help="Patch size with context")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap size to crop")
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()

    fused_model = load_system_for_inference(
        sr_weights_path=args.sr_weights,
        classifier_weights_path=args.class_weights,
        num_classes=args.num_classes,
        sr_scale=args.scale,
    )
    fused_model = fused_model.to(args.device)

    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
        output_dir = args.output_dir or os.path.join(args.input_dir, "sr_results")
        os.makedirs(output_dir, exist_ok=True)

        filenames = sorted(os.listdir(args.input_dir))
        image_paths = [
            os.path.join(args.input_dir, name)
            for name in filenames
            if is_image_file(name)
        ]
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {args.input_dir}")

        for image_path in image_paths:
            base = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base}_sr.png")
            inference_whole_image(
                image_path,
                fused_model,
                args.device,
                patch_size=args.patch_size,
                overlap=args.overlap,
                scale=args.scale,
                output_path=output_path,
            )
    elif args.image:
        inference_whole_image(
            args.image,
            fused_model,
            args.device,
            patch_size=args.patch_size,
            overlap=args.overlap,
            scale=args.scale,
            output_path=args.output,
        )
    else:
        raise ValueError("Provide --image or --input-dir for inference.")
