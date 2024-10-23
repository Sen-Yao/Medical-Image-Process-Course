# Medical-Image-Process-Course

---

这是华中科技大学电信学院闫增强老师 2024 年“医学图像处理”课程的实验代码。

## 实验要求

本实验旨在对给定的皮肤病变图片进行分割。分割方法需涵盖以下所有类型：

- 基础算法（如阈值法、形态学方法等）
- 基于聚类的分割方法
- 基于图的分割方法
- 基于主动轮廓的方法

每种分割方法的特征提取方式需包括以下所有类型，每种特征类型至少需要一种提取方式。基础算法和主动轮廓方法不要求涵盖所有特征类型：

- 强度特征（如灰度、梯度等）
- 纹理特征（如 Gabor 滤波、Laws Kernel 等）
- CNN 特征（采用预训练好的 CNN 模型）

评价指标必须至少涵盖：

- Dice 系数
- 敏感性（Sensitivity）、特异性（Specificity）、准确率（Accuracy）和 AUC（曲线下面积）

选做：使用 ``Data4AI`` 数据集，训练医学图像分割网络以完成分割任务。

## 使用说明

### 安装

首先，克隆项目并创建 conda 环境：

```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
conda env create -f environment.yml
```

### 运行

使用以下命令运行项目：

```
python main.py --model <model name> --feature <feature name> --dataset <dataset name> --config_path <config path>
```

在使用 `UNet` 之前，请在对应的数据集上做训练

```
python train.py --model <model name> --dataset <dataset name> --config_path <config path>
```

#### 参数说明

模型支持以下类型：

- `threshold`: 基于阈值的分割
- `cluster`: 基于 K-means 聚类的分割
- `graph`: 基于图切割的分割
- `contour`: 基于主动轮廓的分割
- `UNet`: 基于 UNet 的分割

特征支持以下类型（仅在使用 `cluster` 或 `graph` 作为分割模型时可用）：

- `RGB`, `HSV`, `LAB`: 使用颜色空间作为特征
- `prewitt`, `sobel`, `canny`: 使用梯度作为特征
- `gabor`, `laws`: 使用纹理作为特征
- `CNN`: 使用预训练的 ResNet 提取特征，仅适用于 `cluster` 分割模型


数据集：

- `Data`: 一个简单的医学图像分割数据集
- `GlaS`: [GlaS](https://arxiv.org/abs/1603.00275)（Gland Segmentation）数据集用于腺体分割任务，包含来自显微镜图像的腺体区域。
- `UDIAT`: UDIAT（Ultrasound Data for Image Analysis Tasks）数据集包含超声图像，主要用于医学图像分割和分析。


你可以在[这里](https://drive.senyao.cloud:444/s/mpEkqMfc2HMPLkr)下载这些数据集 ，并将其放置在如下路径：`/dataset/GIaS`




#### 示例

在 `Data` 数据集上使用 `HSV` 作为特征进行 `cluster` 分割

```
python main.py -m cluster -f HSV -d Data
```

在 `GlaS` 数据集上训练 `UNet`

```
python train.py -m UNet -d GlaS
```

### 贡献

欢迎任何形式的贡献！如果你有建议或发现了问题，请提交 issue 或 pull request。

### 许可证

本项目遵循 MIT 许可证。有关详细信息，请参阅 LICENSE 文件。