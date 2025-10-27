"""
数据加载模块 - 处理CIFAR-10和CIFAR-100数据集
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional
import tarfile
from pathlib import Path

from config import (
    DATASET_DIR, 
    DATA_CONFIG, 
    TRAIN_CONFIG,
    AUGMENT_CONFIG,
    TRAIN_VAL_SPLIT
)


def extract_if_needed(dataset_name: str):
    """如果需要，解压数据集"""
    dataset_dir = Path(DATASET_DIR)
    
    if dataset_name == "cifar10":
        tar_file = dataset_dir / "cifar-10-python.tar.gz"
        extracted_dir = dataset_dir / "cifar-10-batches-py"
    elif dataset_name == "cifar100":
        tar_file = dataset_dir / "cifar-100-python.tar.gz"
        extracted_dir = dataset_dir / "cifar-100-python"
    else:
        return False
    
    # 检查是否需要解压
    if extracted_dir.exists():
        return True
    
    # 检查tar.gz文件是否存在
    if not tar_file.exists():
        print(f"错误: 找不到数据集文件 {tar_file}")
        print(f"请确保文件已下载到 {dataset_dir} 目录")
        return False
    
    # 解压
    print(f"正在解压 {tar_file.name}...")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=dataset_dir)
        print(f"✓ 解压完成: {extracted_dir}")
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def get_cifar_transforms(dataset: str = "cifar10", is_train: bool = True) -> transforms.Compose:
    """
    获取数据增强和预处理变换
    
    Args:
        dataset: 数据集名称 (cifar10 或 cifar100)
        is_train: 是否为训练集
        
    Returns:
        变换组合
    """
    if dataset not in DATA_CONFIG:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    if is_train:
        # 训练集增强
        transform_list = []
        
        if AUGMENT_CONFIG["random_crop"]:
            transform_list.append(transforms.RandomCrop(32, padding=AUGMENT_CONFIG["crop_padding"]))
        
        if AUGMENT_CONFIG["random_horizontal_flip"]:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list.append(transforms.ToTensor())
        
        if AUGMENT_CONFIG["normalize"]:
            if dataset == "cifar10":
                # CIFAR-10 的归一化参数
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ))
            else:  # cifar100
                # CIFAR-100 的归一化参数
                transform_list.append(transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ))
    else:
        # 验证/测试集只进行归一化
        transform_list = [transforms.ToTensor()]
        
        if AUGMENT_CONFIG["normalize"]:
            if dataset == "cifar10":
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ))
            else:
                transform_list.append(transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ))
    
    return transforms.Compose(transform_list)


def load_cifar_dataset(
    dataset: str = "cifar10",
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载CIFAR数据集
    
    Args:
        dataset: 数据集名称 (cifar10 或 cifar100)
        random_seed: 随机种子
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if dataset not in DATA_CONFIG:
        raise ValueError(f"不支持的数据集: {dataset}. 支持: {list(DATA_CONFIG.keys())}")
    
    # 自动解压数据集（如果需要）
    if not extract_if_needed(dataset):
        raise RuntimeError(f"无法解压或访问数据集: {dataset}")
    
    # 使用配置中的比例
    validation_split = TRAIN_VAL_SPLIT["val_ratio"]
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 获取数据集配置
    num_classes = DATA_CONFIG[dataset]["num_classes"]
    dataset_name = DATA_CONFIG[dataset]["dataset_name"]
    
    # 定义训练和测试变换
    train_transform = get_cifar_transforms(dataset, is_train=True)
    test_transform = get_cifar_transforms(dataset, is_train=False)
    
    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR),
        train=True,
        download=False,  # 假设数据集已经下载
        transform=train_transform
    ) if dataset == "cifar10" else datasets.CIFAR100(
        root=str(DATASET_DIR),
        train=True,
        download=False,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR),
        train=False,
        download=False,
        transform=test_transform
    ) if dataset == "cifar10" else datasets.CIFAR100(
        root=str(DATASET_DIR),
        train=False,
        download=False,
        transform=test_transform
    )
    
    # 划分训练集和验证集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    split = int(np.floor(validation_split * num_train))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        sampler=train_sampler,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"],
        drop_last=True
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        sampler=val_sampler,
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"],
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=TRAIN_CONFIG["pin_memory"],
        drop_last=False
    )
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset.upper()}")
    print(f"{'='*60}")
    print(f"训练集样本数: {len(train_indices):,}")
    print(f"验证集样本数: {len(val_indices):,}")
    print(f"测试集样本数: {len(test_dataset):,}")
    print(f"类别数: {num_classes}")
    print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader


def get_class_names(dataset: str) -> list:
    """获取类别名称列表"""
    if dataset == "cifar10":
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset == "cifar100":
        # CIFAR-100有100个细粒度类别
        # 这里返回类别名称（可以后续完善）
        return [f'class_{i}' for i in range(100)]
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载模块...")
    train_loader, val_loader, test_loader = load_cifar_dataset(dataset="cifar10")
    
    # 测试一个batch
    for images, labels in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"图像范围: [{images.min():.2f}, {images.max():.2f}]")
        break

