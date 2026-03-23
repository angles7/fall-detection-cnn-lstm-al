# Author: Jayson SHI
# Email: 2040420809@qq.com
# Date: 2026-03-23
# Description: Real-time fall detection
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from typing import Dict, List, Union, Optional
from datetime import datetime

#  模型结构定义（必须与训练时完全一致） 
class CNNLSTM(nn.Module):
    """CNN + LSTM 视频分类模型"""
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
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.squeeze(-1).squeeze(-1)
        c_out = self.fc_in(c_out)
        lstm_in = c_out.view(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        last_out = lstm_out[:, -1, :]
        logits = self.fc_out(last_out)
        return logits


#  基类 
class BaseVideoAgent:
    """视频分析智能体基类，定义预测接口"""
    def __init__(self, seed: int = None):
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int = 42) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        pass

    def predict(self, video_path: str) -> Dict:
        raise NotImplementedError

    def close(self) -> None:
        pass


#  跌倒检测智能体（适配训练代码生成的全局最优模型） 
class FallDetectionAgent(BaseVideoAgent):

    def __init__(self, seed: int = None, model_dir: str = "."):
        super(FallDetectionAgent, self).__init__(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.model_dir = model_dir
        self._check_required_files()
        self._load_config()
        self._load_model()

        # 类别信息（与训练代码一致：0=跌倒，1=未跌倒）
        self.class_info = {
            0: {
                "name": "fall",
                "label": "跌倒",
                "description": "检测到跌倒事件，请关注",
                "emoji": "⚠️",
                "color": "red"
            },
            1: {
                "name": "not_fall",
                "label": "未跌倒",
                "description": "视频中未检测到跌倒行为",
                "emoji": "✅",
                "color": "green"
            }
        }

        # 图像预处理（必须与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"✓ 跌倒检测Agent初始化完成")
        print(f"  模型: {self.model_path}")
        print(f"  序列长度: {self.seq_len}")
        print(f"  类别: {self.class_names}")

    def _check_required_files(self):
        """检查模型文件是否存在（使用训练生成的全局最优模型）"""
        self.model_path = os.path.join(self.model_dir, "best_cnn_lstm_global.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请先运行训练代码生成全局最优模型。"
            )

    def _load_config(self):
        """加载配置文件（可选），若不存在则使用训练时的默认值"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.seq_len = config.get("sequence_length", 16)
            self.frame_size = config.get("frame_size", 224)
            self.num_classes = config.get("num_classes", 2)
            self.class_names = config.get("class_names", ["fall", "not_fall"])
        else:
            print("⚠️ 未找到配置文件，使用训练时的默认参数：seq_len=16, frame_size=224, classes=2")
            self.seq_len = 16
            self.frame_size = 224
            self.num_classes = 2
            self.class_names = ["fall", "not_fall"]

    def _load_model(self):
        """加载模型权重"""
        try:
            self.model = CNNLSTM(num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"✓ 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def _load_frames_from_folder(self, folder_path: str) -> List[Image.Image]:
        """从文件夹中读取所有图片并按名称排序"""
        if not os.path.isdir(folder_path):
            raise ValueError(f"路径不是文件夹: {folder_path}")

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(valid_ext)]
        if len(files) == 0:
            raise ValueError(f"文件夹中没有图片: {folder_path}")

        files.sort()  # 假设帧名称有序
        images = []
        for f in files:
            img_path = os.path.join(folder_path, f)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        return images

    def _uniform_sample(self, images: List[Image.Image], seq_len: int) -> List[Image.Image]:
        """均匀采样指定数量的帧"""
        total = len(images)
        if total < seq_len:
            # 帧数不足时，重复最后一帧填充
            indices = list(range(total)) + [total - 1] * (seq_len - total)
        else:
            indices = np.linspace(0, total - 1, seq_len, dtype=int).tolist()
        return [images[i] for i in indices]

    def preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """将图片列表转换为模型输入张量 [1, seq_len, C, H, W]"""
        frames = []
        for img in images:
            tensor = self.transform(img)  # [C, H, W]
            frames.append(tensor)
        clip = torch.stack(frames)  # [seq_len, C, H, W]
        clip = clip.unsqueeze(0)    # [1, seq_len, C, H, W]
        return clip.to(self.device)

    def predict(self, video_path: str) -> Dict:
        """
        对视频帧目录进行跌倒检测
        :param video_path: 包含视频帧图片的文件夹路径
        :return: 预测结果字典
        """
        try:
            #  加载图片
            images = self._load_frames_from_folder(video_path)
            #  均匀采样
            sampled = self._uniform_sample(images, self.seq_len)
            #  预处理
            input_tensor = self.preprocess(sampled)
            #  推理
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)  # [1, num_classes]
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            # 5. 解析结果
            class_info = self.class_info.get(pred_class, {
                "name": "unknown",
                "label": "未知",
                "description": "无法识别",
                "emoji": "❓"
            })

            # 所有类别的概率
            all_probs = {}
            for i, name in enumerate(self.class_names):
                all_probs[name] = float(probs[0, i].item())

            # 构建返回字典
            result = {
                "status": "success",
                "video_path": video_path,
                "frame_count": len(images),
                "sampled_frames": self.seq_len,
                "prediction": {
                    "class": pred_class,
                    "label": class_info["label"],
                    "name": class_info["name"],
                    "confidence": confidence,
                    "probabilities": all_probs,
                    "description": class_info["description"],
                    "emoji": class_info["emoji"]
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return result

        except Exception as e:
            return {
                "status": "error",
                "video_path": video_path,
                "message": str(e),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def print_result(self, result: Dict, verbose: bool = True):
        """打印预测结果到控制台"""
        if result["status"] != "success":
            print(f"❌ 预测失败: {result.get('message', '未知错误')}")
            return

        print("\n" + "=" * 60)
        print(f"{result['prediction']['emoji']} 跌倒检测结果")
        print("=" * 60)
        print(f"📁 视频帧目录: {result['video_path']}")
        print(f"📊 总帧数: {result['frame_count']} | 采样帧数: {result['sampled_frames']}")

        print(f"\n🎯 预测类别: {result['prediction']['label']} ({result['prediction']['name']})")
        print(f"📈 置信度: {result['prediction']['confidence']:.2%}")
        print(f"📝 描述: {result['prediction']['description']}")

        if verbose and "probabilities" in result["prediction"]:
            print("\n📊 各类别概率:")
            for name, prob in result["prediction"]["probabilities"].items():
                bar = "█" * int(prob * 20)
                print(f"  {name}: {prob:.2%} {bar}")

        print(f"\n⏰ 预测时间: {result['timestamp']}")
        print("=" * 60)

    def reset(self) -> None:
        super().reset()

    def close(self) -> None:
        super().close()


#  测试函数 
def test_examples():
    """使用示例帧文件夹进行测试（需提前准备）"""
    print("=" * 60)
    print("跌倒检测系统 - 测试模式（全局最优模型）")
    print("=" * 60)

    # 请根据实际路径修改测试数据
    test_cases = [
        "data/test/fall/sample1",      # 跌倒样本（标签0）
        "data/test/not_fall/sample2",  # 未跌倒样本（标签1）
    ]

    try:
        agent = FallDetectionAgent(seed=42)

        for path in test_cases:
            if not os.path.isdir(path):
                print(f"\n⚠️ 测试路径不存在，跳过: {path}")
                continue
            print(f"\n📁 测试: {path}")
            result = agent.predict(path)
            agent.print_result(result, verbose=True)

        agent.close()
        print("\n✓ 测试完成！")
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.exit(test_examples())
    else:
        print("跌倒检测 Agent 使用说明:")
        print("=" * 60)
        print("在 Python 代码中导入并使用：")
        print("  from agent import FallDetectionAgent")
        print("  agent = FallDetectionAgent()")
        print("  result = agent.predict('path/to/frames')")
        print("  agent.print_result(result)")
        print("\n或直接运行测试：")
        print("  python agent.py")
