// UI 引用
const ui = {
    bar: document.getElementById('model-progress'),
    statusTitle: document.getElementById('status-title'),
    statusDetail: document.getElementById('status-detail'),
    log: document.getElementById('log'),
    btn: document.getElementById('upload-btn'),
    input: document.getElementById('file-upload'),
    dropArea: document.getElementById('drop-area'),
    downloadBtn: document.getElementById('download-btn'),
    canvasIn: document.getElementById('canvas-input'),
    canvasOut: document.getElementById('canvas-output')
};

// 初始化 Worker
const worker = new Worker('worker.js');

// ================= Worker 消息处理 =================
worker.onmessage = function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'log':
            log(data);
            break;
        case 'progress':
            updateProgress(data.percent, data.title, data.detail);
            break;
        case 'status':
            if (data === 'ready') {
                ui.btn.classList.add('active');
                updateProgress(100, "System Ready", "Waiting for image");
            }
            break;
        case 'error':
            updateProgress(100, "Error", "See logs", true);
            log(`❌ Error: ${data}`);
            ui.btn.classList.add('active');
            break;
        case 'tile-result':
            drawTile(data);
            break;
        case 'done':
            updateProgress(100, "Done!", "Processing Complete");
            ui.btn.classList.add('active');
            // 显示下载按钮
            ui.downloadBtn.style.display = 'inline-flex';
            ui.downloadBtn.classList.add('active');
            break;
    }
};

// ================= UI 逻辑 =================

function log(msg) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerText = `> ${msg}`;
    ui.log.appendChild(div);
    ui.log.scrollTop = ui.log.scrollHeight;
}

function updateProgress(percent, title, detail, isError = false) {
    if (isError) ui.bar.classList.add('error');
    else ui.bar.classList.remove('error', 'success');
    
    if (percent >= 100 && !isError) ui.bar.classList.add('success');
    
    ui.bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    if (title) ui.statusTitle.innerText = title;
    if (detail) ui.statusDetail.innerText = detail;
}

function drawTile({ buffer, x, y, width, height }) {
    const u8Clamped = new Uint8ClampedArray(buffer);
    const imgData = new ImageData(u8Clamped, width, height);
    const ctx = ui.canvasOut.getContext('2d');
    ctx.putImageData(imgData, x, y);
}

// ================= 核心：文件处理逻辑 =================

function processFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    // 重置 UI 状态
    ui.btn.classList.remove('active');
    ui.downloadBtn.style.display = 'none'; // 隐藏下载按钮
    updateProgress(0, "Processing", "Loading image...");
    
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
        // 1. 绘制原图
        ui.canvasIn.width = img.width;
        ui.canvasIn.height = img.height;
        const ctxIn = ui.canvasIn.getContext('2d', { willReadFrequently: true });
        ctxIn.drawImage(img, 0, 0);

        // 2. 准备输出画布
        const scale = 2; 
        ui.canvasOut.width = img.width * scale;
        ui.canvasOut.height = img.height * scale;
        const ctxOut = ui.canvasOut.getContext('2d');
        ctxOut.fillStyle = "#111"; // 黑色背景，避免透明图问题
        ctxOut.fillRect(0, 0, ui.canvasOut.width, ui.canvasOut.height);

        // 3. 获取像素数据发送给 Worker
        const imageData = ctxIn.getImageData(0, 0, img.width, img.height);
        const buffer = imageData.data.buffer; 

        worker.postMessage({
            type: 'run',
            data: {
                width: img.width,
                height: img.height,
                buffer: buffer
            }
        }, [buffer]); 
    };
}

// ================= 事件监听 (点击、拖放、下载) =================

// 1. 点击上传
ui.input.addEventListener('change', (e) => processFile(e.target.files[0]));

// 2. 拖放上传
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    ui.dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

ui.dropArea.addEventListener('dragover', () => ui.dropArea.classList.add('highlight'));
ui.dropArea.addEventListener('dragleave', () => ui.dropArea.classList.remove('highlight'));
ui.dropArea.addEventListener('drop', (e) => {
    ui.dropArea.classList.remove('highlight');
    const dt = e.dataTransfer;
    const files = dt.files;
    processFile(files[0]);
});

// 3. 点击下载
ui.downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'upscaled_image.png';
    link.href = ui.canvasOut.toDataURL('image/png');
    link.click();
});

// 启动 Worker 初始化
worker.postMessage({ type: 'init' });
