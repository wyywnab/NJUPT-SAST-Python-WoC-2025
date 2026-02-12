importScripts('https://fastly.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/ort.min.js');

// 修复 WASM 路径
ort.env.wasm.wasmPaths = "https://fastly.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";

// ================= 配置区域 =================
const CONFIG = {
    models: {
        classifier: './web_models/classifier_packed.onnx',
        experts: [
            './web_models/expert_0_packed.onnx',
            './web_models/expert_1_packed.onnx',
            './web_models/expert_2_packed.onnx'
        ]
    },
    scale: 2,
    patchSize: 112,
    overlap: 16,
};

// 序号到文本的映射配置
const CLASS_NAMES = [
    "Anime / Illustration", // Index 0
    "Real World / Photo",   // Index 1
    "Text / Screenshot"     // Index 2
];

const sessions = { classifier: null, experts: [] };

// ================= 消息监听 =================
onmessage = async (e) => {
    const { type, data } = e.data;
    try {
        if (type === 'init') await initModels();
        else if (type === 'run') await runInference(data);
    } catch (err) {
        console.error(err);
        postMessage({ type: 'error', data: err.message });
    }
};

// ================= 1. 模型加载 =================
async function initModels() {
    postMessage({ type: 'log', data: '🚀 Worker initializing...' });
    
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4; 
    ort.env.wasm.proxy = false;
    ort.env.wasm.simd = true;

    try {
        sessions.classifier = await loadModelWithProgress(CONFIG.models.classifier, "Classifier");
        for (let i = 0; i < CONFIG.models.experts.length; i++) {
            const url = CONFIG.models.experts[i];
            const session = await loadModelWithProgress(url, `Expert ${i+1}`);
            sessions.experts.push(session);
        }
        postMessage({ type: 'status', data: 'ready' });
        postMessage({ type: 'log', data: '✅ Worker Ready & Models Loaded' });
    } catch (e) {
        throw new Error(`Model Load Failed: ${e.message}`);
    }
}

async function loadModelWithProgress(url, name) {
    postMessage({ type: 'progress', data: { percent: 0, title: `Downloading ${name}`, detail: 'Connecting...' } });
    
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Fetch failed: ${response.statusText}`);

    const total = parseInt(response.headers.get('content-length') || '0');
    const reader = response.body.getReader();
    let received = 0;
    let chunks = [];

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        
        if (total) {
            const percent = (received / total) * 100;
            if (received % (1024 * 1024) < 10000) { 
                postMessage({ type: 'progress', data: { percent, title: `Downloading ${name}`, detail: `${(received/1024/1024).toFixed(1)}MB` } });
            }
        }
    }

    let buffer = new Uint8Array(received);
    let pos = 0;
    for (let chunk of chunks) {
        buffer.set(chunk, pos);
        pos += chunk.length;
    }

    postMessage({ type: 'progress', data: { percent: 100, title: `Compiling ${name}`, detail: 'WASM Init...' } });
    
    return await ort.InferenceSession.create(buffer.buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
    });
}

// ================= 2. 推理逻辑 =================
async function runInference({ width, height, buffer }) {
    const inputData = new Uint8ClampedArray(buffer);

    // --- Step 1: 分类 ---
    postMessage({ type: 'log', data: '🔍 Analyzing Scene...' });
    const classIdx = await runClassifier(inputData, width, height);
    
    // 获取文本名称
    const className = CLASS_NAMES[classIdx] || `Unknown Class (${classIdx})`;
    postMessage({ type: 'log', data: `🎯 Detected: [${className}] -> Using Expert_${classIdx}` });

    // --- Step 2: 分块超分 ---
    const session = sessions.experts[classIdx];
    await runTiledSR(inputData, width, height, session);

    postMessage({ type: 'done' });
}

async function runClassifier(data, w, h) {
    const float32Data = new Float32Array(3 * 224 * 224);
    const scaleX = w / 224;
    const scaleY = h / 224;

    for (let y = 0; y < 224; y++) {
        for (let x = 0; x < 224; x++) {
            const srcX = Math.floor(x * scaleX);
            const srcY = Math.floor(y * scaleY);
            const srcIdx = (srcY * w + srcX) * 4;
            
            const dstIdx = y * 224 + x;
            float32Data[dstIdx] = (data[srcIdx] - 127.5) / 127.5;
            float32Data[dstIdx + 50176] = (data[srcIdx + 1] - 127.5) / 127.5;
            float32Data[dstIdx + 100352] = (data[srcIdx + 2] - 127.5) / 127.5;
        }
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
    const results = await sessions.classifier.run({ input: tensor });
    const logits = results.logits.data;
    
    let maxIdx = 0, maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
        if (logits[i] > maxVal) { maxVal = logits[i]; maxIdx = i; }
    }
    return maxIdx;
}

async function runTiledSR(fullData, w, h, session) {
    const P = CONFIG.patchSize;
    const O = CONFIG.overlap;
    const S = CONFIG.scale;
    const stride = P - 2 * O;

    const ySteps = [];
    for (let y = 0; y < h; y += stride) ySteps.push(y);
    const xSteps = [];
    for (let x = 0; x < w; x += stride) xSteps.push(x);
    const totalTiles = ySteps.length * xSteps.length;

    const inputBuffer = new Float32Array(3 * P * P);
    const cropStart = O * S;
    const cropEnd = (P - O) * S;
    const cropLen = cropEnd - cropStart;
    const patchOutDim = P * S;

    let processed = 0;

    for (let y of ySteps) {
        for (let x of xSteps) {
            fillInputBuffer(fullData, inputBuffer, x, y, P, O, w, h);
            
            const tensor = new ort.Tensor('float32', inputBuffer, [1, 3, P, P]);
            const results = await session.run({ input: tensor });
            const outputFloat = results.output.data;

            const tileBuffer = new Uint8ClampedArray(cropLen * cropLen * 4);
            fillTileBuffer(outputFloat, tileBuffer, cropStart, cropEnd, patchOutDim);

            postMessage({
                type: 'tile-result',
                data: {
                    buffer: tileBuffer.buffer,
                    x: x * S,
                    y: y * S,
                    width: cropLen,
                    height: cropLen
                }
            }, [tileBuffer.buffer]);

            processed++;
            if (processed % 2 === 0) {
                 postMessage({ 
                     type: 'progress', 
                     data: { percent: (processed/totalTiles)*100, title: 'Upscaling...', detail: `${Math.round((processed/totalTiles)*100)}%` } 
                 });
            }
        }
    }
}

function fillInputBuffer(src, target, startX, startY, P, O, W, H) {
    const readY = startY - O;
    const readX = startX - O;
    const area = P * P;
    let ptr = 0;
    const inv255 = 0.00392156862;

    for (let py = 0; py < P; py++) {
        let gy = readY + py;
        if (gy < 0) gy = 0; else if (gy >= H) gy = H - 1;
        const rowOffset = gy * W;

        for (let px = 0; px < P; px++) {
            let gx = readX + px;
            if (gx < 0) gx = 0; else if (gx >= W) gx = W - 1;
            
            const idx = (rowOffset + gx) << 2;
            target[ptr] = src[idx] * inv255;
            target[ptr + area] = src[idx+1] * inv255;
            target[ptr + area*2] = src[idx+2] * inv255;
            ptr++;
        }
    }
}

function fillTileBuffer(floatData, targetU8, start, end, fullDim) {
    const stride = fullDim * fullDim;
    let ptr = 0;
    
    for (let y = start; y < end; y++) {
        const rowOffset = y * fullDim;
        for (let x = start; x < end; x++) {
            const idx = rowOffset + x;
            
            let r = floatData[idx];
            let g = floatData[idx + stride];
            let b = floatData[idx + stride * 2];

            if (r<0) r=0; else if(r>1) r=1;
            if (g<0) g=0; else if(g>1) g=1;
            if (b<0) b=0; else if(b>1) b=1;

            targetU8[ptr++] = (r * 255) | 0;
            targetU8[ptr++] = (g * 255) | 0;
            targetU8[ptr++] = (b * 255) | 0;
            targetU8[ptr++] = 255;
        }
    }
}
