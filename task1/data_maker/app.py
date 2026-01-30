import os
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__, template_folder='./')

# ================= 配置区域 =================
INPUT_FOLDER = 'source_images'  # 你的原始图片文件夹
OUTPUT_FOLDER = 'dataset_output' # 结果保存位置
HR_SIZE = 256   # 高清块大小
LR_SIZE = 128   # 低清块大小
TARGET_2K = 2048 # 框选区域将被缩放至的长边大小
# ===========================================

# 确保输出目录存在
os.makedirs(os.path.join(OUTPUT_FOLDER, 'HR'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'LR'), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    # 获取所有 jpg, png, jpeg 图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    # 只返回文件名，按名称排序
    files = sorted([os.path.basename(f) for f in files])
    return jsonify(files)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(INPUT_FOLDER, filename)

@app.route('/api/process', methods=['POST'])
def process_image():
    data = request.json
    filename = data['filename']
    crop_data = data['crop'] # {x, y, width, height}

    img_path = os.path.join(INPUT_FOLDER, filename)
    
    try:
        with Image.open(img_path) as img:
            # 1. 裁剪 (注意：前端传来的可能是浮点数，需转int)
            left = int(crop_data['x'])
            top = int(crop_data['y'])
            right = left + int(crop_data['width'])
            bottom = top + int(crop_data['height'])
            
            cropped_img = img.crop((left, top, right, bottom))

            # 2. 压缩/缩放到 2K (以长边为准，保持比例)
            w, h = cropped_img.size
            if w > h:
                new_w = TARGET_2K
                new_h = int(h * (TARGET_2K / w))
            else:
                new_h = TARGET_2K
                new_w = int(w * (TARGET_2K / h))
            
            # 使用高质量重采样
            resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 3. 切块处理
            count = 0
            # 简单的网格切分，步长等于块大小（无重叠）
            for y in range(0, new_h - HR_SIZE + 1, HR_SIZE):
                for x in range(0, new_w - HR_SIZE + 1, HR_SIZE):
                    # 提取 HR 块 (256x256)
                    box = (x, y, x + HR_SIZE, y + HR_SIZE)
                    hr_patch = resized_img.crop(box)

                    # 生成 LR 块 (128x128)
                    lr_patch = hr_patch.resize((LR_SIZE, LR_SIZE), Image.Resampling.BICUBIC)

                    # 保存文件名：原文件名_序号.png
                    base_name = os.path.splitext(filename)[0]
                    save_name = f"{base_name}_{count}.png"

                    hr_patch.save(os.path.join(OUTPUT_FOLDER, 'HR', save_name))
                    lr_patch.save(os.path.join(OUTPUT_FOLDER, 'LR', save_name))
                    count += 1

        return jsonify({"status": "success", "patches_created": count})

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(f"请把图片放入 '{INPUT_FOLDER}' 文件夹")
    print("启动服务中... 请在浏览器访问 http://127.0.0.1:5000")
    app.run(debug=True)
