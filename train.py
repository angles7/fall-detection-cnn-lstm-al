# Author: Jayson SHI
# Email: 2040420809@qq.com
# Date: 2026-03-23
# Description: Real-time fall detection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

#  超参数 
SEQUENCE_LENGTH = 16          # 每个视频采样的帧数
FRAME_SIZE = 224              # 图像resize大小
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 2               # 跌倒 / 未跌倒
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 主动学习相关参数
DATA_ROOT = 'data'                     # 包含 fall/ 和 not_fall/ 的根目录
ACTIVE_LEARNING_ROUNDS = 3              # 主动学习轮数
TRAIN_RATIO = 0.6                       # 训练集比例

#  数据集定义（支持直接传入样本列表） 
class VideoFrameDataset(Dataset):
    def __init__(self, sequence_length=16, transform=None, samples=None, root_dir=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        if samples is not None:
            # 直接使用传入的样本列表
            self.samples = samples
        elif root_dir is not None:
            # 从根目录扫描
            classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            for cls in classes:
                cls_path = os.path.join(root_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for video_folder in os.listdir(cls_path):
                    video_path = os.path.join(cls_path, video_folder)
                    if not os.path.isdir(video_path):
                        continue
                    frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
                    if len(frames) >= sequence_length:
                        self.samples.append((video_path, self.class_to_idx[cls]))
        else:
            raise ValueError("必须提供 samples 或 root_dir 参数")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
        total_frames = len(frames)
        # 均匀采样 sequence_length 帧
        indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        images = []
        for i in indices:
            img_path = os.path.join(video_path, frames[i])
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        # 堆叠为 (seq_len, C, H, W)
        images = torch.stack(images)
        return images, label
    
#  数据预处理 
transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  模型定义（CNN + LSTM） 
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, lstm_num_layers=2):
        super(CNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.fc_in = nn.Linear(512, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.3 if lstm_num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, C, H, W]
        batch_size, seq_len, C, H, W = x.size()
        # 合并 batch 和 seq_len 以便通过 CNN
        c_in = x.view(batch_size * seq_len, C, H, W)          
        c_out = self.cnn(c_in)                                 
        c_out = c_out.squeeze(-1).squeeze(-1)                  
        c_out = self.fc_in(c_out)                              
        # 恢复序列维度
        lstm_in = c_out.view(batch_size, seq_len, -1)          
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)              
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]                           
        logits = self.fc_out(last_out)                          
        return logits

#  辅助函数 
def collect_all_samples(root_dir, sequence_length=16):
    """从fall和not_fall目录收集所有满足帧数要求的样本"""
    all_samples = []
    # 固定类别映射：fall -> 0, not_fall -> 1
    class_to_idx = {'fall': 0, 'not_fall': 1}
    for cls_name, label in class_to_idx.items():
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            print(f"警告：目录 {cls_path} 不存在，跳过。")
            continue
        for video_folder in os.listdir(cls_path):
            video_path = os.path.join(cls_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
            if len(frames) >= sequence_length:
                all_samples.append((video_path, label))
    return all_samples

def split_samples(samples, train_ratio, seed=None):
    """按比例随机划分训练集和测试集"""
    if seed is not None:
        random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(samples) * train_ratio)
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    return train_samples, test_samples

def predict_sample(model, video_path, transform):
    """预测单个视频样本的类别"""
    model.eval()
    # 获取所有帧并采样
    frames = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.jpg','.png'))])
    total_frames = len(frames)
    indices = np.linspace(0, total_frames-1, SEQUENCE_LENGTH, dtype=int)
    images = []
    for i in indices:
        img_path = os.path.join(video_path, frames[i])
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)
    # 堆叠并添加batch维度
    video_tensor = torch.stack(images).unsqueeze(0).to(DEVICE)  # [1, seq_len, C, H, W]
    with torch.no_grad():
        outputs = model(video_tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

def find_error_samples(model, test_samples, transform):
    """返回测试集中预测错误的样本列表"""
    error_samples = []
    for video_path, true_label in tqdm(test_samples, desc="Finding errors"):
        pred_label = predict_sample(model, video_path, transform)
        if pred_label != true_label:
            error_samples.append((video_path, true_label))
    return error_samples

def update_split(train_samples, test_samples, error_samples):
    # 将错误样本从测试集中移除
    test_set = set(test_samples)
    error_set = set(error_samples)
    new_test = list(test_set - error_set)
    # 将错误样本加入训练集
    new_train = train_samples + error_samples
    # 从训练集中（排除错误样本）随机抽取等量样本移入测试集
    candidate_pool = [s for s in new_train if s not in error_set]  # 错误样本保留在训练集中
    num_to_move = len(error_samples)
    if len(candidate_pool) < num_to_move:
        raise RuntimeError("训练集候选池不足，无法移动样本！")
    moved = random.sample(candidate_pool, num_to_move)
    # 从训练集中移除这些样本
    new_train = [s for s in new_train if s not in moved]
    # 加入测试集
    new_test.extend(moved)

    return new_train, new_test

#  训练与验证函数 
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc='Training')
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=correct/total)
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=correct/total)
    return total_loss / len(loader), correct / total

#  主程序 
def main():
    #  收集所有样本
    print("Collecting all samples...")
    all_samples = collect_all_samples(DATA_ROOT, SEQUENCE_LENGTH)
    print(f"Total samples: {len(all_samples)}")

    #  初始划分（训练:测试 = 6:4）
    train_samples, test_samples = split_samples(all_samples, TRAIN_RATIO, seed=42)
    print(f"Initial split: train={len(train_samples)}, test={len(test_samples)}")

    # 全局最佳模型跟踪
    global_best_acc = 0.0
    global_best_model_path = 'best_cnn_lstm_global.pth'

    #  主动学习循环
    for round_idx in range(ACTIVE_LEARNING_ROUNDS):
        print(f"\n========== Active Learning Round {round_idx+1} ==========")

        # 根据当前划分创建数据集和数据加载器
        train_dataset = VideoFrameDataset(sequence_length=SEQUENCE_LENGTH, transform=transform, samples=train_samples)
        test_dataset = VideoFrameDataset(sequence_length=SEQUENCE_LENGTH, transform=transform, samples=test_samples)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # 初始化模型、损失函数、优化器
        model = CNNLSTM(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_acc = 0.0
        best_model_path = f'best_cnn_lstm_round{round_idx+1}.pth'

        # 内层训练循环
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = validate(model, test_loader, criterion)
            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 保存当前轮次最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model for round {round_idx+1} saved with val_acc: {best_acc:.4f}")

            # 保存全局最佳模型
            if val_acc > global_best_acc:
                global_best_acc = val_acc
                torch.save(model.state_dict(), global_best_model_path)
                print(f"New global best model saved with val_acc: {global_best_acc:.4f}")

        # 加载本轮最佳模型，在测试集上寻找错误样本
        print("\nLoading best model to find error samples...")
        model.load_state_dict(torch.load(best_model_path))
        error_samples = find_error_samples(model, test_samples, transform)
        print(f"Found {len(error_samples)} error samples in test set.")

        if len(error_samples) == 0:
            print("No errors found. Stopping active learning.")
            break

        # 更新训练集和测试集划分
        train_samples, test_samples = update_split(train_samples, test_samples, error_samples)
        print(f"After round {round_idx+1}: train={len(train_samples)}, test={len(test_samples)}")

    print("\nActive learning finished.")
    print(f"Global best accuracy: {global_best_acc:.4f} (model saved as {global_best_model_path})")

if __name__ == "__main__":
    main()
