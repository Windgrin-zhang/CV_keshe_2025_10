"""
模型构建模块
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from config import MODEL_DIR, DATA_CONFIG, TRAIN_STRATEGY, MODEL_CONFIG


def build_model(model_name: str, dataset: str, device: torch.device):
    """
    构建ResNet模型
    
    Args:
        model_name: 模型名称 (resnet18/34/50/101/152)
        dataset: 数据集名称 (cifar10/cifar100)
        device: 设备
        
    Returns:
        model: PyTorch模型
    """
    # 获取类别数
    num_classes = DATA_CONFIG[dataset]["num_classes"]
    
    # 创建模型
    model_factory = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }
    
    if model_name not in model_factory:
        raise ValueError(f"不支持的模型: {model_name}")
    
    model = model_factory[model_name](pretrained=False)
    
    # 修改最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 加载预训练权重（如果存在）
    if TRAIN_STRATEGY["use_pretrained"]:
        config = MODEL_CONFIG[model_name]
        weights_path = MODEL_DIR / config["weights_file"]
        
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            # 过滤掉最后一层（因为类别数不同）
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            model.load_state_dict(state_dict, strict=False)
            
            # 重新初始化最后一层
            model.fc.weight.data.normal_(0, 0.01)
            model.fc.bias.data.zero_()
    
    # 移动到设备
    model = model.to(device)
    
    # 冻结backbone（如果设置了）
    if TRAIN_STRATEGY["freeze_backbone"]:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    return model


def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

