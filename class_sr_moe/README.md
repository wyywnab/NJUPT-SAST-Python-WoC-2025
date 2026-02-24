# 多任务学习 - Adaptive SR

## 简介

- 目标：实现不同图片场景的分类超分

- 方式：通过先对图片进行分类，选择专门的专家网络进行超分

- 分类：共3类
  - *Anime / Illustration (anime)*

  - *Real World / Photo (landscape)*

  - *Text / Screenshot (text)*

## 复现

- 训练：修改config.yaml后运行train.py

- 评估：

  ```bash
  python evaluate.py \
  	--class-data <path/to/classification_dataset>
  	--sr-data <path/to/sr_dataset>
  	--class-weights <path/to/classifier_weights>
  	--sr-weights <path/to/sr(fused)_weights>
  	--scale 2
  	--device <cuda_or_cpu>
  	--class-batch 32
  	--sr-batch 8
  	--num_workers 4
  	--seed <random_seed>
  	--out-csv <path/to/results_file>
  ```

- 推理：

  ```bash
  python inference.py \
  	--image <path/to/input/single_image>
  	--output <path/to/output/single_image>
  	--input-dir <path/to/folder/images>
  	--output-dir <path/to/output/folder>
  	--class-weights <path/to/classifier_weights>
  	--sr-weights <path/to/sr(fused)_weights>
      --scale 2
  	--device <cuda_or_cpu>
  	--patch-size <patch_size_to_split_image>
  	--overlap <overlap_size_to_crop>
  ```

## Web

Web demo：[v2_web](v2_web)

在线推理：[https://wyywn.site/projects/adasr_web/](https://wyywn.site/projects/adasr_web/)

## 数据

- 来源：

  - *anime*：网络收集的高质量图片，带有丰富细节

  - *landscape*：DIV2K节选


  - *text*：网页、PPT、PDF截图

### 1. 分类

- 总量：

  | 类别/数据集 | train | val  | test |
  | ----------- | ----- | ---- | ---- |
  | anime       | 719   | 90   | 90   |
  | landscape   | 716   | 89   | 90   |
  | text        | 627   | 78   | 79   |

- 处理：
  1. 将原图统一缩放到224×224，并确保原图长宽比不超过2.5:1
  2. 按8:1:1分割训练、验证、测试集

### 2. 超分

- 总量：

  | 类别/数据集 | train | val  | test |
  | ----------- | ----- | ---- | ---- |
  | anime       | 4640  | 580  | 580  |
  | landscape   | 4640  | 580  | 580  |
  | text        | 4640  | 580  | 580  |

- 处理：
  1. 原图切分为224×224的小块
  2. 通过计算每个小块图片的评分筛选有实际内容的图片，作为HR图
  3. 将HR图片缩放到112×112，作为LR图

## 网络结构

*SRNet、ClassNet与task1、task2的网络结构定义完全相同，省略*

```mermaid
graph TD
    Input[("Input Image x<br>(B, C, H, W)")] --> CheckGT{gt_label provided?}
    GT[("gt_label (B,)")] --> CheckGT

    CheckGT -- Yes<br>(Train SR Experts) --> UseGT["gt_label<br>↓<br>indices"]
    CheckGT -- No<br>(Inference) --> InferencePath["Inference: use classifier"]

    subgraph InferencePath [Classification]
        direction TB
        Classifier["Classifier (frozen)"] --> Logits["class_logits<br>(B, num_classes)"]
        Logits --> ArgMax["argmax(dim=1)"] --> SelectedIndices["selected_indices (B,)"]
    end

    UseGT --> SelectedIndices

    SelectedIndices --> GroupByClass["Group batch indices<br> by unique class"]

    subgraph ExpertProcessing [Expert SR Networks]
        Expert0["SRExpert 0<br>(scale_factor=sr_scale)"] --> Out0
        Expert1["SRExpert 1<br>(scale_factor=sr_scale, num_res_blocks=32)"] --> Out1
        Expert2["SRExpert 2<br>(scale_factor=sr_scale)"] --> Out2
    end

    GroupByClass -- "class 0 indices" --> Expert0
    GroupByClass -- "class 1 indices" --> Expert1
    GroupByClass -- "class 2 indices" --> Expert2

    Out0["Output subset 0"] --> Combine
    Out1["Output subset 1"] --> Combine
    Out2["Output subset 2"] --> Combine

    Combine["Combine outputs<br>original batch order"] --> FinalOutput["final_output<br>(B, C, H\*scale, W\*scale)"]

    Logits --> OutputLogits["class_logits<br>(if inference)"]

    FinalOutput --> Return
    OutputLogits --> Return
    Return[("Return<br>(final_output, class_logits)")]
```

*说明：v2版本对SRExpert 1的num_res_blocks数量翻倍处理为32（v1版本为16）*

## 指标

| 组别             | 准确率 |
| ---------------- | ------ |
| 分类集/test      | 96.14% |
| 超分集/thumbnail | 95.25% |

| 组别      | 类别      | PSNR (dB) | SSIM   | 总计 |
| --------- | --------- | --------- | ------ | ---- |
| sr_oracle | overall   | 33.49     | 94.76% | 1740 |
| sr_oracle | anime     | 34.07     | 95.54% | 580  |
| sr_oracle | landscape | 30.09     | 89.50% | 580  |
| sr_oracle | text      | 36.30     | 99.25% | 580  |
| sr_mixed  | overall   | 33.48     | 94.75% | 1740 |
| sr_mixed  | anime     | 34.09     | 95.54% | 580  |
| sr_mixed  | landscape | 30.06     | 89.46% | 580  |
| sr_mixed  | text      | 36.29     | 99.25% | 580  |
| sr_random | overall   | 31.89     | 93.82% | 1740 |
| sr_random | anime     | 33.20     | 94.88% | 580  |
| sr_random | landscape | 29.27     | 88.06% | 580  |
| sr_random | text      | 33.20     | 98.51% | 580  |

*说明：oracle - 只超分（理想专家）， mixed - 网络整体（实际分类+专家）， random - 随机专家*

与Task1的对比：

| 任务/专家 | 数据集     | PSNR (dB) | SSIM   | 总数 |
| --------- | ---------- | --------- | ------ | ---- |
| Task1     | Task1/test | 43.25     | 0.9954 | 630  |
| 多任务/0  | Task1/all  | 41.18     | 0.9833 | 4197 |
| 多任务/1  | Task1/all  | 42.01     | 0.9857 | 4197 |
| 多任务/2  | Task1/all  | 39.07     | 0.9758 | 4197 |

分析：

- **分类器**
  - 分类器在测试集上的准确率达到 **96.14%**，在超分任务缩略图上也达到了 **95.25%** 的高准确率。这说明模型能够有效区分三种不同类型的图像，为后续选择适合的超分专家网络提供了可靠依据。
- **专家网络**
  - 从 `sr_oracle`（理想专家）和 `sr_mixed`（实际分类+专家）的结果看，两者的 PSNR 和 SSIM 指标非常接近，整体 PSNR 分别为 **33.49** 和 **33.48**，SSIM 分别为 **94.76%** 和 **94.75%**，说明分类器几乎没有引入性能损失，能够准确地为每类图像分配最合适的专家网络。
  - 对比 `sr_random`（随机选择专家）的结果，其整体 PSNR 降至 **31.89**，SSIM 降至 **93.82%**，明显低于前两者。这进一步说明，专门化的专家网络对不同类型图像的超分效果确实优于通用或随机分配的网络。
- **不同类别**
  - 在所有设置中，`text` 类别的超分效果最好（PSNR 高达 **36.30**，SSIM 达 **99.25%**），这可能是因为文本图像具有清晰的边缘和结构，易于重建。
  - `landscape` 类别的超分效果相对较差（PSNR 约 **30.06**，SSIM 约 **89.46%**），说明自然图像细节丰富、纹理复杂，超分难度较大。
  - `anime` 类别的效果居中（PSNR 约 **34.09**，SSIM 约 **95.54%**），说明其具有一定的结构性和纹理特征，但仍能较好地重建。
- **为何Task1与多任务的PSNR指标相差大**
  - 根据与Task1的对比，PSNR 差距（约 10dB）主要源于数据本身的复杂度。
  - 证据：多任务模型在处理其自身复杂数据时仅为 **33.5 dB**，但当直接迁移到Task 1的简单数据集上，未经过微调最佳专家（Expert 1）即跑出了 **42.01 dB** 的高分。
  - 结论：Task1的高指标主要得益于图像的大面积纯色背景和低频信息，任务难度较低；而多任务面对自然纹理和复杂场景，任务难度高。

