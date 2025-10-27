"""
最小化测试 - 测试数据集加载和模型构建
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

import torch
import torch.nn as nn
import torchvision.models as models
from config import MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG, MODEL_DIR

print("="*60)
print("最小化测试")
print("="*60)

# 1. 测试数据集加载
print("\n1. 测试数据集加载:")
try:
    from data_loader import load_cifar_dataset
    train_loader, val_loader, test_loader = load_cifar_dataset("cifar10")
    print(f"✓ 数据集加载成功")
    print(f"  训练batches: {len(train_loader)}")
    print(f"  验证batches: {len(val_loader)}")
    print(f"  测试batches: {len(test_loader)}")
    
    # 测试一个batch
    for images, labels in train_loader:
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        break
except Exception as e:
    print(f"✗ 数据集加载失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试模型构建
print("\n2. 测试模型构建:")
try:
    model_name = "resnet18"
    num_classes = 10
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print(f"✓ 模型创建成功: {model_name}")
    print(f"  类别数: {num_classes}")
    
    # 检查预训练权重
    weights_path = MODEL_DIR / f"{model_name}_imagenet.pth"
    if weights_path.exists():
        print(f"  ✓ 找到预训练权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        print(f"    权重文件大小: {weights_path.stat().st_size / (1024*1024):.1f} MB")
        
        # 尝试加载（不加载最后一层）
        model_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(model_dict, strict=False)
        print(f"    权重已加载（除最后一层）")
    else:
        print(f"  ✗ 预训练权重不存在: {weights_path}")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
except Exception as e:
    print(f"✗ 模型构建失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试前向传播
print("\n3. 测试前向传播:")
try:
    # 创建一个随机输入
    dummy_input = torch.randn(2, 3, 32, 32)  # batch_size=2, CIFAR图像大小
    output = model(dummy_input)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成！")
print("="*60)

