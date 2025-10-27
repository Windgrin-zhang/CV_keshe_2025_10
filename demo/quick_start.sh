#!/bin/bash
# 快速开始脚本

echo "========================================"
echo "ResNet CIFAR 训练系统 - 快速开始"
echo "========================================"

# 进入项目根目录
cd "$(dirname "$0")/.."

echo ""
echo "步骤1: 运行最小化测试"
echo "----------------------------------------"
python demo/test_minimal.py

echo ""
echo "步骤2: 运行完整训练演示（1个epoch）"
echo "----------------------------------------"
python demo/demo_train.py

echo ""
echo "========================================"
echo "✓ 所有测试完成！"
echo "========================================"
echo ""
echo "现在可以运行正式训练:"
echo "  cd script"
echo "  python train.py --model resnet18 --dataset cifar10 --epochs 100"
echo ""
echo "或者使用快速脚本:"
echo "  cd script"
echo "  ./run_train.sh --model resnet18 --dataset cifar10 --epochs 10"
echo ""

