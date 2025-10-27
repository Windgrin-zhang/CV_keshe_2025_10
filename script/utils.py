"""
工具函数模块 - 用于可视化和辅助功能
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from pathlib import Path
import json


def plot_training_curves(metrics_file: Path, save_path: Path):
    """
    绘制训练曲线 - 准确率和损失
    
    Args:
        metrics_file: 指标JSON文件
        save_path: 保存路径
    """
    # 读取指标
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        return
    
    # 提取数据
    epochs = [m['epoch'] for m in metrics]
    train_acc = [m['train_acc'] for m in metrics]
    val_acc = [m['val_acc'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    val_loss = [m['val_loss'] for m in metrics]
    
    # 创建2个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：准确率
    axes[0].plot(epochs, train_acc, label='Train Acc', marker='o', markersize=3)
    axes[0].plot(epochs, val_acc, label='Val Acc', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：损失
    axes[1].plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    axes[1].plot(epochs, val_loss, label='Val Loss', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lr_curve(metrics_file: Path, save_path: Path):
    """
    绘制学习率曲线
    
    Args:
        metrics_file: 指标JSON文件
        save_path: 保存路径
    """
    # 读取指标
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        return
    
    # 提取数据
    epochs = [m['epoch'] for m in metrics]
    lr = [m['learning_rate'] for m in metrics]
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lr, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_memory_usage(metrics_file: Path, save_path: Path):
    """
    绘制内存使用曲线
    
    Args:
        metrics_file: 指标JSON文件
        save_path: 保存路径
    """
    # 读取指标
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics:
        return
    
    # 提取数据
    epochs = [m['epoch'] for m in metrics]
    memory = [m['memory_mb'] for m in metrics]
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, memory, marker='o', markersize=3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage During Training')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

