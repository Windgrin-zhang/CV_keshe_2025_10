"""
配置文件 - 包含所有训练超参数和路径设置
"""

import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

# 创建输出目录（YOLO风格）
OUTPUT_DIR.mkdir(exist_ok=True)

# 数据配置
DATA_CONFIG = {
    "cifar10": {
        "num_classes": 10,
        "dataset_name": "cifar10",
    },
    "cifar100": {
        "num_classes": 100,
        "dataset_name": "cifar100",
    }
}

# 模型配置
MODEL_CONFIG = {
    "resnet18": {
        "weights_file": "resnet18_imagenet.pth",
    },
    "resnet34": {
        "weights_file": "resnet34_imagenet.pth",
    },
    "resnet50": {
        "weights_file": "resnet50_imagenet.pth",
    },
    "resnet101": {
        "weights_file": "resnet101_imagenet.pth",
    },
    "resnet152": {
        "weights_file": "resnet152_imagenet.pth",
    }
}

# 数据分割比例
TRAIN_VAL_SPLIT = {
    "train_ratio": 0.8,  # 训练集占80%
    "val_ratio": 0.2,     # 验证集占20%
}

# 训练超参数（从网上找到的推荐值）
TRAIN_CONFIG = {
    "batch_size": 128,
    "num_epochs": 100,
    "learning_rate": 0.1,  # CIFAR常用0.1
    "momentum": 0.9,
    "weight_decay": 5e-4,  # 权重衰减，防止过拟合
    "gamma": 0.1,          # 学习率衰减因子
    "milestones": [60, 80], # 学习率衰减里程碑
    "num_workers": 4,
    "pin_memory": True,
    "device": "cuda",
}

# 数据增强配置
AUGMENT_CONFIG = {
    "random_crop": True,
    "random_horizontal_flip": True,
    "normalize": True,
    "crop_padding": 4,
}

# 训练策略配置
TRAIN_STRATEGY = {
    "use_pretrained": True,
    "freeze_backbone": False,
}

# 可视化配置
VISUALIZE_CONFIG = {
    "save_plots": True,
    "plot_format": "png",
}

# 保存配置
SAVE_CONFIG = {
    "save_freq": 10,  # 每N个epoch保存一次
    "save_best": True,  # 保存最佳模型
}
