# NJUPT-SAST-Python-WoC-2025

此仓库为*学术向--多任务学习*的一个实现。

- Task1 - 超分

  - 网络架构：主要为16层卷积与残差
  - 数据与目标：手机拍摄的白板笔记超分
  - 指标：测试集平均 43.25 dB

- Task2 - CIFAR分类

  - 网络架构：主要为8层卷积与残差
  - 指标：测试集平均准确率 86.13%

- 多任务学习 - 适应性超分

  - 目标：实现不同图片场景（3类）的适应性超分

  - 方式：通过先对图片进行分类，选择专门的专家网络进行超分

  - 网络架构：大致与task1相同，有略微改动

    - *Anime / Illustration*：主要为16层卷积与残差
    - *Real World / Photo*：主要为(16`->`)32层卷积与残差
    - *Text / Screenshot*：主要为16层卷积与残差

  - 数据集：

    - *Anime / Illustration (anime)*：网络收集的高质量图片，带有丰富细节

    - *Real World / Photo (landscape)*：DIV2K节选

    - *Text / Screenshot (text)*：网页、PPT、PDF截图

  - 指标：

| 组别             | 准确率 |
| ---------------- | ------ |
| 分类集 test      | 96.14% |
| 超分集 thumbnail | 95.25% |

| 组别      | 类别      | PSNR  | SSIM   | 总计 |
| --------- | --------- | ----- | ------ | ---- |
| sr_oracle | overall   | 33.49 | 94.76% | 1740 |
| sr_oracle | anime     | 34.07 | 95.54% | 580  |
| sr_oracle | landscape | 30.09 | 89.50% | 580  |
| sr_oracle | text      | 36.30 | 99.25% | 580  |
| sr_mixed  | overall   | 33.48 | 94.75% | 1740 |
| sr_mixed  | anime     | 34.09 | 95.54% | 580  |
| sr_mixed  | landscape | 30.06 | 89.46% | 580  |
| sr_mixed  | text      | 36.29 | 99.25% | 580  |
| sr_random | overall   | 31.89 | 93.82% | 1740 |
| sr_random | anime     | 33.20 | 94.88% | 580  |
| sr_random | landscape | 29.27 | 88.06% | 580  |
| sr_random | text      | 33.20 | 98.51% | 580  |

*说明：oracle - 只评测超分表现（确定专家）， mixed - 评测适应性超分（即先分类再超分）， random - 评测随机专家超分*



每个Task的单独README将在之后提交

