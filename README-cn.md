# 跌倒检测 (Fall Detection) 训练与使用指南（主动学习版）
本项目基于 **CNN + LSTM** 实现视频级别的跌倒/未跌倒二分类，并引入**主动学习**机制，通过迭代挖掘困难样本提升模型性能。代码使用 PyTorch 框架，从视频帧序列中提取空间特征并建模时序依赖，可用于监控视频中的跌倒事件检测。

## 特点
- 使用预训练的 ResNet18 提取每帧的空间特征，节省训练时间并提升性能。
- 通过 LSTM 捕捉帧间的时序依赖，适用于视频分类任务。
- 主动学习策略：每轮训练结束后，用当前最佳模型找出测试集中的错误样本，将其移入训练集，并从训练集中随机抽取等量样本补入测试集，保持 6:4 的比例不变。多轮迭代后，模型更关注困难样本，泛化能力更强。
- 数据集自动划分：无需手动划分训练/验证集，程序从 `data/fall` 和 `data/not_fall` 中读取所有样本，并按 6:4 随机划分为训练集和测试集（固定随机种子保证可重复性）。
- 全局最优模型保存：训练过程中不仅保存每轮的最佳模型（`best_cnn_lstm_roundX.pth`），还保存所有轮次中验证准确率最高的全局最优模型（`best_cnn_lstm_global.pth`），方便最终部署。
- 学习率自适应调整（ReduceLROnPlateau），避免陷入局部最优。
- 完整的训练/验证流程，支持 GPU 加速（自动选择设备）。
- 提供配套的推理脚本 `agent.py`，封装模型加载、预处理和结果解析，方便集成。

## 环境要求
- Python 3.7+
- PyTorch 1.7+ (推荐 1.10+)
- torchvision (与 PyTorch 版本匹配)
- numpy
- Pillow
- tqdm

## 安装依赖
```bash
pip install torch torchvision numpy pillow tqdm
```
如果使用 GPU，请确保安装的 PyTorch 版本与 CUDA 版本匹配，具体请参考 [PyTorch 官网](https://pytorch.org/)。

## 数据集准备
### 目录结构规范
请将数据集按照以下目录结构组织：
```
data/
├── fall/                     # 跌倒类别
│   ├── video_001/            # 视频001的帧图片
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   ├── ...
│   ├── video_002/
│   └── ...
└── not_fall/                  # 未跌倒类别
    ├── video_101/
    └── ...
```
类别名称：文件夹名即为类别名（`fall`、`not_fall`），程序会自动映射标签（`fall=0`, `not_fall=1`）。
视频文件夹：每个视频的所有帧放在一个独立的文件夹中，文件夹名称任意。
帧图片：放在视频文件夹内，支持 `.jpg`、`.png`、`.jpeg`、`.bmp` 格式。

### 帧图像要求
图片命名应能够按顺序排序（例如使用 `frame_0001.jpg`、`frame_0002.jpg` 等），程序使用 `sorted()` 对文件名进行排序，因此建议使用**固定位数的数字编号**（如 `%06d`）。
所有图片应为 RGB 三通道，训练时会自动转换为 RGB。
图片尺寸不限，训练时会统一 resize 到 `FRAME_SIZE × FRAME_SIZE`（默认 224×224）。

### 数据集划分说明
程序会自动从 `data/fall` 和 `data/not_fall` 中收集所有满足帧数要求的视频文件夹，然后按 6:4 随机划分为训练集和测试集（固定随机种子 `seed=42` 保证可重复性）。因此，您**无需**手动创建 `train` 和 `val` 目录。

### 数据量建议
- 每个类别至少包含几十个视频，以保证模型泛化能力。
- 如果数据量较少，可考虑增加数据增强（本项目未包含，可根据需要自行添加）。
## 配置文件与超参数
所有超参数在代码开头的“超参数”部分定义，可根据需要修改。

### 超参数说明
| 参数 | 默认值 | 说明 |
| `SEQUENCE_LENGTH` | 16 | 每个视频均匀采样的帧数。如果视频帧数不足该值，该视频会被跳过。 |
| `FRAME_SIZE` | 224 | 输入图片 resize 的尺寸（宽=高）。 |
| `BATCH_SIZE` | 8 | 批大小，根据 GPU 显存调整。 |
| `EPOCHS` | 50 | 每轮主动学习内部训练的轮数。 |
| `LEARNING_RATE` | 1e-4 | 初始学习率。 |
| `NUM_CLASSES` | 2 | 类别数（跌倒/未跌倒）。 |
| `ACTIVE_LEARNING_ROUNDS` | 3 | 主动学习轮数。每轮结束后会重新划分数据集。 |
| `TRAIN_RATIO` | 0.6 | 训练集比例（其余为测试集）。 |
| `DEVICE` | 自动选择 | 优先使用 cuda，否则使用 cpu。 |
### 数据集路径
代码中默认数据根目录为 `DATA_ROOT = 'data'`，即从 `data/fall` 和 `data/not_fall` 读取。如需修改，请更改：
```python
DATA_ROOT = 'your_data_path'
```

## 训练模型
### 启动训练
确保数据集准备就绪后，直接运行：
```bash
python train.py
```
训练过程将：
1. 收集所有样本，按 6:4 随机划分训练集和测试集。
2. 进入主动学习循环，每轮内进行 `EPOCHS` 次训练迭代。
3. 每轮训练结束后，用该轮最佳模型找出测试集中的错误样本，更新数据集划分。
4. 每轮内保存该轮最佳模型（`best_cnn_lstm_roundX.pth`），同时保存全局最优模型（`best_cnn_lstm_global.pth`）。
5. 打印每轮的训练/验证损失和准确率，并显示当前轮的错误样本数及更新后的数据集大小。
### 训练过程监控
- 使用 `tqdm` 显示进度条，可直观看到当前 batch 的损失和累计准确率。
- 每个 epoch 结束后打印训练损失、准确率和验证损失、准确率。
- 主动学习每轮结束后显示错误样本数量及新的训练/测试集规模。
### 模型保存
每轮最佳模型：`best_cnn_lstm_round1.pth`、`best_cnn_lstm_round2.pth` ...
全局最佳模型：`best_cnn_lstm_global.pth`（所有轮次中验证准确率最高的模型）
模型保存的是 `state_dict`，不包含网络结构，因此推理时需要重新定义相同的模型类。
### 恢复训练
如果你想从某个 checkpoint 继续训练，可以在初始化模型后加载权重，然后继续迭代。由于主动学习涉及数据集变化，恢复训练需要手动调整轮次和数据集状态，建议完整运行脚本。

## 模型结构详解
模型 `CNNLSTM` 包含四个主要部分：
1. CNN 特征提取器
使用预训练的 ResNet18（去掉最后的全连接层和平均池化层之后的卷积部分），输出形状为 `(batch, 512, 1, 1)` 的特征图。我们通过 `squeeze` 将其变为 `(batch, 512)`。
```python
resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]  # 去掉最后的全连接层
self.cnn = nn.Sequential(*modules)
```
2. 特征映射层
将 512 维的 CNN 特征映射到 256 维，供 LSTM 使用。
```python
self.fc_in = nn.Linear(512, 256)
```
3. LSTM 时序建模
- 输入大小：256
- 隐藏层大小：512
- 层数：2
- 是否双向：否（单向）
- Dropout：0.3（仅当层数>1时有效）
- batch_first=True
LSTM 接收形状 `(batch, seq_len, 256)` 的序列，输出所有时间步的隐藏状态 `(batch, seq_len, 512)` 以及最后一个时间步的隐藏状态和细胞状态。
4. 分类层
取 LSTM 最后一个时间步的输出（`lstm_out[:, -1, :]`），通过全连接层输出 logits（2 维）。
```python
self.fc_out = nn.Linear(lstm_hidden_size, num_classes)
```

### 前向传播流程
1. 输入 `x` 形状：`(batch, seq_len, C, H, W)`
2. 合并 batch 和 seq_len：`(batch * seq_len, C, H, W)`
3. 通过 CNN：`(batch * seq_len, 512, 1, 1)` → 压缩为 `(batch * seq_len, 512)`
4. 通过 `fc_in`：`(batch * seq_len, 256)`
5. 恢复序列维度：`(batch, seq_len, 256)`
6. 输入 LSTM：`(batch, seq_len, 256)` → 输出 `(batch, seq_len, 512)`
7. 取最后一个时间步：`(batch, 512)`
8. 通过 `fc_out`：`(batch, num_classes)`

## 评估与推理
训练完成后，可以使用配套的推理脚本 `agent.py`（位于同目录下）对新的视频帧文件夹进行预测。该脚本默认加载全局最优模型 `best_cnn_lstm_global.pth`。

### 使用配套推理脚本 agent.py
`agent.py` 提供了 `FallDetectionAgent` 类，用法如下：
```python
from agent import FallDetectionAgent
# 初始化 agent（自动加载 best_cnn_lstm_global.pth）
agent = FallDetectionAgent(model_dir='.')  # model_dir 为模型所在目录
# 对视频帧文件夹进行预测
result = agent.predict("path/to/video_frames")
# 打印美观的结果
agent.print_result(result, verbose=True)
```

### 预测结果格式
`result` 是一个字典，包含以下字段：
`status`: "success" 或 "error"
`video_path`: 输入路径
`frame_count`: 文件夹中的帧总数
`sampled_frames`: 实际采样的帧数（等于 `SEQUENCE_LENGTH`）
`prediction`:
  `class`: 预测类别索引（0 表示跌倒，1 表示未跌倒）
  `label`: 类别中文标签（“跌倒”/“未跌倒”）
   `name`: 类别英文名（“fall”/“not_fall”）
  `confidence`: 置信度
  `probabilities`: 所有类别的概率
  `description`: 描述信息
  `emoji`: 表情符号
`timestamp`: 预测时间戳

### 单独使用模型进行预测
如果不想使用 agent，也可以直接加载模型进行预测：
```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
# 定义模型（与训练时相同）
model = CNNLSTM(num_classes=2)
model.load_state_dict(torch.load('best_cnn_lstm_global.pth'))
model.eval()
model.to(device)
# 预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def predict_video(frames_folder):
    images = []
    for f in sorted(os.listdir(frames_folder)):
        if f.lower().endswith(('.jpg','.png')):
            img = Image.open(os.path.join(frames_folder, f)).convert('RGB')
            images.append(img)
    # 均匀采样16帧
    indices = np.linspace(0, len(images)-1, 16, dtype=int)
    sampled = [images[i] for i in indices]
    # 预处理
    tensor_list = [transform(img) for img in sampled]
    clip = torch.stack(tensor_list).unsqueeze(0).to(device)  # [1,16,C,H,W]
    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()
```

## 自定义数据集完整流程
1. 收集视频：获取原始监控视频。
2. 提取帧：使用 FFmpeg 等工具将视频按固定帧率（如 30fps）提取为图片，并统一命名（例如 `frame_%06d.jpg`）。
   ```bash
   ffmpeg -i input.mp4 -vf fps=30 frames/frame_%06d.jpg
   ```
3. 组织目录：将每个视频的帧文件夹分别放入 `data/fall/` 或 `data/not_fall/` 下（根据类别）。
4. 修改超参数（可选）：如调整 `SEQUENCE_LENGTH`、`BATCH_SIZE` 等。
5. 运行训练：直接执行 `python train.py`。
6. 评估：训练完成后，可使用测试集（程序内部保留）的准确率作为参考，也可用 agent 对新样本进行测试。
7. 部署：使用训练好的全局最优模型进行推理。

## 性能优化建议
增大 batch size：在显存允许的情况下，增大 batch size 可以加速训练并可能提高稳定性。
学习率调度：已使用 `ReduceLROnPlateau`，可根据验证损失自动调整。
早停：可以监控验证损失，若连续多个 epoch 不下降则提前终止训练（可自行添加）。
混合精度训练：使用 `torch.cuda.amp` 可减少显存占用并加速训练（需要 PyTorch 1.6+），可自行集成。
多 GPU 训练：可以使用 `nn.DataParallel` 包装模型，但需注意 batch 分配。

## 引用与参考
[PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
LSTM: [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

本项目代码仅供学习和研究使用，如需商用请自行评估。
**This code was designed by Jayson SHI，For any questions regarding this code, please contact:2040420809@qq.com**
