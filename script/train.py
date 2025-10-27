"""
主训练脚本 - YOLO风格
"""

import torch
import argparse
from pathlib import Path
import sys

from config import (
    OUTPUT_DIR, TRAIN_CONFIG, MODEL_CONFIG, DATA_CONFIG
)
from data_loader import load_cifar_dataset
from model import build_model, count_parameters
from trainer import Trainer
from utils import plot_training_curves, plot_lr_curve, plot_memory_usage


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练ResNet模型在CIFAR数据集上')
    
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=list(MODEL_CONFIG.keys()),
                        help='模型名称')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学习率')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 更新配置
    TRAIN_CONFIG["num_epochs"] = args.epochs
    TRAIN_CONFIG["batch_size"] = args.batch_size
    TRAIN_CONFIG["learning_rate"] = args.lr
    
    # 设备
    device = torch.device(TRAIN_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / f"{args.model}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 加载数据集
    print("加载数据集...")
    train_loader, val_loader, test_loader = load_cifar_dataset(args.dataset)
    
    # 构建模型
    print(f"\n构建模型: {args.model}")
    model = build_model(args.model, args.dataset, device)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name=args.model,
        dataset=args.dataset,
        output_dir=output_dir
    )
    
    # 开始训练
    trainer.train()
    
    # 绘制可视化图表
    print("\n生成可视化图表...")
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        plot_training_curves(metrics_file, output_dir / "training_curves.png")
        plot_lr_curve(metrics_file, output_dir / "lr_curve.png")
        plot_memory_usage(metrics_file, output_dir / "memory_usage.png")
        print("可视化图表已保存到输出目录")
    
    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print(f"  - best.pt: 最佳模型")
    print(f"  - last.pt: 最新模型")
    print(f"  - training.log: 训练日志")
    print(f"  - metrics.json: 训练指标")
    print(f"  - *.png: 可视化图表")


if __name__ == "__main__":
    main()
