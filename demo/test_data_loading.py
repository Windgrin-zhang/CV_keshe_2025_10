"""
测试数据加载模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

import torch
from torchvision import datasets
from pathlib import Path

# 测试CIFAR数据集的实际情况
DATASET_DIR = Path("/home/alex/VScode/课程设计/dataset")

print("="*60)
print("测试数据集加载")
print("="*60)

# 检查文件结构
print("\n1. 检查dataset目录:")
files = list(DATASET_DIR.glob("*"))
for f in files:
    print(f"  - {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

# 检查是否有解压后的文件夹
print("\n2. 检查是否有解压后的数据:")
for dataset_name in ["cifar-10-batches-py", "cifar-100-python"]:
    data_dir = DATASET_DIR / dataset_name
    if data_dir.exists():
        print(f"  ✓ 找到解压后的数据: {data_dir}")
        subfiles = list(data_dir.glob("*"))
        print(f"    包含 {len(subfiles)} 个文件")
    else:
        print(f"  ✗ 未找到: {data_dir}")

# 尝试加载CIFAR-10
print("\n3. 尝试加载CIFAR-10:")
try:
    cifar10_dir = DATASET_DIR / "cifar-10-batches-py"
    if cifar10_dir.exists():
        test_dataset = datasets.CIFAR10(
            root=str(DATASET_DIR),
            train=True,
            download=False,
            transform=None
        )
        print(f"  ✓ 成功加载！数据集大小: {len(test_dataset)}")
        print(f"    类别数: {len(test_dataset.classes)}")
        print(f"    类别: {test_dataset.classes[:5]}")
    else:
        print("  ✗ cifar-10-batches-py目录不存在")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")

# 检查tar.gz文件
print("\n4. 检查压缩文件:")
tar_files = list(DATASET_DIR.glob("*.tar.gz"))
for tar_file in tar_files:
    print(f"  - {tar_file.name}")

print("\n" + "="*60)

