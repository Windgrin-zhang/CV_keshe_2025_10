#!/bin/bash
# 快速训练脚本

# 设置参数
MODEL="resnet18"
DATASET="cifar10"
EPOCHS=100
BATCH_SIZE=128
LR=0.001
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "开始训练"
echo "========================================"
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LR"
echo "设备: $DEVICE"
echo "========================================"

# 运行训练
python train.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --device $DEVICE

echo "训练完成！"

