"""
模型评估脚本 - 评估已训练的模型并生成详细报告
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MODEL_DIR, CHECKPOINT_DIR, RESULT_DIR,
    MODEL_CONFIG, DATA_CONFIG, TRAIN_CONFIG
)
from data_loader import load_cifar_dataset, get_class_names


def load_model_from_checkpoint(checkpoint_path, model_name, dataset, device='cuda'):
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点文件路径
        model_name: 模型名称
        dataset: 数据集名称
        device: 设备
        
    Returns:
        加载的模型
    """
    # 创建模型结构
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=False)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=False)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 修改最后一层
    num_classes = DATA_CONFIG[dataset]["num_classes"]
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"已加载模型: {checkpoint_path}")
    print(f"训练轮数: {checkpoint['epoch']}")
    print(f"最佳验证准确率: {checkpoint['best_val_acc']:.2f}%")
    
    return model


def evaluate_model(model, dataloader, device, class_names=None):
    """
    详细评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        class_names: 类别名称列表
        
    Returns:
        (accuracy, loss, predictions, labels, probabilities)
    """
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 获取预测和概率
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            running_loss += loss.item()
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # 计算准确率
    accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)
    avg_loss = running_loss / len(dataloader)
    
    return accuracy, avg_loss, all_predictions, all_labels, all_probabilities


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置字体大小
    plt.rcParams['font.size'] = 14  # 基础字体大小
    plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签
    plt.rcParams['xtick.labelsize'] = 12  # x轴刻度
    plt.rcParams['ytick.labelsize'] = 12  # y轴刻度
    plt.rcParams['legend.fontsize'] = 14  # 图例
    plt.rcParams['figure.titlesize'] = 18  # 标题
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 12})  # 注释字体大小
    plt.ylabel('真实标签', fontsize=18)
    plt.xlabel('预测标签', fontsize=18)
    plt.title('混淆矩阵', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失曲线')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('准确率曲线')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=list(MODEL_CONFIG.keys()),
                        help='模型名称')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='数据集名称')
    parser.add_argument('--checkpoint', type=str,
                        default='best',  # 'best', 'last', 或具体路径
                        help='要加载的检查点 (best/last/路径)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 确定检查点路径
    if args.checkpoint == 'best':
        checkpoint_path = CHECKPOINT_DIR / f'{args.model}_{args.dataset}_best.pth'
    elif args.checkpoint == 'last':
        checkpoint_path = CHECKPOINT_DIR / f'{args.model}_{args.dataset}_last.pth'
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return
    
    # 加载模型
    model = load_model_from_checkpoint(
        checkpoint_path, args.model, args.dataset, device
    )
    
    # 加载数据集
    _, _, test_loader = load_cifar_dataset(args.dataset)
    
    # 获取类别名称
    class_names = get_class_names(args.dataset)
    
    # 评估模型
    print("\n正在评估模型...")
    accuracy, loss, predictions, labels, probabilities = evaluate_model(
        model, test_loader, device, class_names
    )
    
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"测试集准确率: {accuracy:.2f}%")
    print(f"测试集损失: {loss:.4f}")
    print(f"{'='*60}\n")
    
    # 分类报告
    print("详细分类报告:")
    print(classification_report(
        labels, predictions,
        target_names=class_names if len(class_names) <= 10 else None
    ))
    
    # 如果数据集类别不多，绘制混淆矩阵
    if args.plot and len(class_names) <= 20:
        cm_path = RESULT_DIR / f'{args.model}_{args.dataset}_confusion_matrix.png'
        plot_confusion_matrix(labels, predictions, class_names, cm_path)
    
    # 如果有训练历史，绘制曲线
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'history' in checkpoint:
        history = checkpoint['history']
        plot_training_history(history, RESULT_DIR / f'{args.model}_{args.dataset}_history.png')
    
    # 保存评估结果
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'test_accuracy': float(accuracy),
        'test_loss': float(loss),
        'num_samples': len(labels),
    }
    
    result_path = RESULT_DIR / f'{args.model}_{args.dataset}_evaluation.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存: {result_path}")


if __name__ == "__main__":
    main()

