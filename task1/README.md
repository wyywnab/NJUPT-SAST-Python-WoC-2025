# Task1 - 超分

## 复现

- 训练：修改config.yaml后运行train.py

- 评估：修改evaluate.py中exp_folder和model_path并运行

- 测试图片：

  ```bash
  python test.py \
      --input <path/to/image_or_dir> \
      --output <path/to/output/dir> \
      --model <path/to/model.pth>
  ```

## 数据

- 来源：手机拍摄的白板笔记

- 处理：

  1. 手动将高清图片切除边缘空白部分

  2. 把图片压缩/缩放到长边长2048，使用LANCZOS算法

  3. 自动分割成小块图像（256×256）作为HR图片
  4. 把HR图片缩小到128×128，使用BICUBIC算法，作为LR图片
  5. 自动划分数据为70%训练，15%验证，15%测试

## 网络结构

### Residual Block

```mermaid
graph TD
    Input[("Input x")] --> Conv1["Conv2d<br>kernel=3x3<br>padding=1, bias=True<br>channels → channels"]
    Conv1 --> ReLU["ReLU<br>(inplace=True)"]
    ReLU --> Conv2["Conv2d<br>kernel=3x3<br>padding=1, bias=True<br>channels → channels"]
    Conv2 --> Mul["× 0.1"]
    Mul --> Add["Add"]
    Input --> Add
    Add --> Output[("Output")]
```

### UpsampleBlock

``` mermaid
graph TD
    A["Input <br>(B, C, H, W)"] --> B["Conv2d<br> kernel=3x3, padding=1<br>out (B, C×4, H, W)"]
    B --> C["PixelShuffle(2)<br>out (B, C, H×2, W×2)"]
    C --> D["ReLU (inplace=True)"]
    D --> E["Output<br>(B, C, H×2, W×2)"]
```



### SRNet

``` mermaid
graph TD
    Input[("Input<br>(H x W x C)")] --> HeadConv["Head Conv2d<br>kernel=3x3<br>num_channels → n_feats"]
    HeadConv --> HeadOut["Head Output (x)"]
    HeadOut --> Body["Body<br>16 × <b>ResidualBlock</b>"]
    Body --> MidConv["Conv2d<br>kernel=3x3<br>n_feats → n_feats"]
    MidConv --> AddNode["Add<br>res + x<br>(skip connection)"]
    HeadOut --> AddNode
    AddNode --> Upsample["UpsampleBlock<br>scale_factor=2<br>Conv + PixelShuffle + ReLU"]
    Upsample --> TailConv["Tail Conv2d<br>kernel=3x3<br>n_feats → num_channels"]
    TailConv --> Output[("Output<br>(H\*scale × W\*scale × C)")]
```

## 指标

测试集平均PSNR 43.25 dB