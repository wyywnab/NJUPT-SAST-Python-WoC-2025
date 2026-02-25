# NJUPT-SAST-Python-WoC-2025

## 简介

方向：*学术向--多任务学习*

- Task1 - 超分 [README](task1/README.md)

  - 网络架构：主要为16层卷积与残差
  - 数据与目标：手机拍摄的白板笔记超分
  - 指标：测试集平均PSNR 43.25 dB

- Task2 - CIFAR分类 [README](task2/README.md)

  - 网络架构：主要为8层卷积与残差
  - 指标：测试集平均准确率 86.13%

- 多任务学习 - Adaptive SR [README](class_sr_moe/README.md)

  - 目标：实现不同图片场景（3类）的分类超分

  - 方式：通过先对图片进行分类，选择专门的专家网络进行超分

  - 网络架构：大致与task1相同，有略微改动

    - *Anime / Illustration*：主要为16层卷积与残差
    - *Real World / Photo*：主要为(16`->`)32层卷积与残差
    - *Text / Screenshot*：主要为16层卷积与残差

  - 数据集：

    - *Anime / Illustration (anime)*：网络收集的高质量图片，带有丰富细节

    - *Real World / Photo (landscape)*：DIV2K节选

    - *Text / Screenshot (text)*：网页、PPT、PDF截图

  - 指标：分类+混合专家总PSNR 33.48dB，SSIM 94.75%

## 复现

安装依赖

```bash
pip install -r requirements.txt 
```

数据集：[sast_python_woc_dataset.7z](https://wyywn.site/onemanager/public/sast_python_woc_dataset.7z)

注：每个task均为独立项目，需要在每个项目project目录下单独运行

### 环境信息

task1 & task2：

| 项目        | 版本         |
| ----------- | ------------ |
| PyTorch     | 2.7.1+cu118  |
| torchvision | 0.22.1+cu118 |
| CUDA        | 11.8         |
| GPU Driver  | 575.57.08    |
| GPU         | Tesla P4     |

多任务 v2 class：

| 项目        | 版本                               |
| ----------- | ---------------------------------- |
| PyTorch     | 2.10.0+cu130                       |
| torchvision | 0.25.0+cu130                       |
| CUDA        | 13.0                               |
| GPU Driver  | 591.74                             |
| GPU         | NVIDIA GeForce RTX 5060 Laptop GPU |

多任务 v2 fused：

| 项目        | 版本                      |
| ----------- | ------------------------- |
| PyTorch     | 2.8.0+cu128               |
| torchvision | 0.23.0+cu128              |
| CUDA        | 12.8                      |
| GPU Driver  | 580.105.08                |
| GPU         | NVIDIA GeForce RTX 4090 D |

