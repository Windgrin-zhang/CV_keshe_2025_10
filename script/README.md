# CIFAR ResNet训练系统 - YOLO风格

## 📁 目录结构

```
script/
├── train.py          # 主训练脚本（只包含主函数）
├── config.py          # 配置管理
├── data_loader.py     # 数据加载
├── model.py           # 模型构建
├── trainer.py         # 训练逻辑
├── logger.py          # 日志记录（YOLO风格）
├── utils.py           # 可视化和工具函数
├── evaluate.py        # 模型评估
└── requirements.txt   # 依赖包
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd script
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练（100 epochs）
python train.py

# 自定义参数
python train.py --model resnet50 --dataset cifar10 --epochs 100 --batch_size 128 --lr 0.1

# 快速测试
python train.py --epochs 10 --batch_size 64
```

### 3. 输出文件

所有结果保存在 `output/` 目录：

```
output/resnet18_cifar10/
├── best.pt              # 最佳模型（YOLO风格）
├── last.pt              # 最新模型
├── training.log         # 训练日志
├── metrics.json         # 训练指标
├── training_curves.png  # 准确率和损失曲线
├── lr_curve.png         # 学习率曲线
└── memory_usage.png     # 内存使用曲线
```

## 📊 超参数配置

### 推荐超参数（从网上找到的最佳实践）

#### CIFAR-10
- `learning_rate`: 0.1
- `batch_size`: 128
- `momentum`: 0.9
- `weight_decay`: 5e-4
- `milestones`: [60, 80]
- `gamma`: 0.1

#### CIFAR-100
- `learning_rate`: 0.1
- `batch_size`: 128
- `momentum`: 0.9
- `weight_decay`: 5e-4
- `milestones`: [60, 80]
- `gamma`: 0.1

在 `config.py` 中可以修改这些参数。

## 📈 使用示例

### 示例1: 训练ResNet18 on CIFAR-10

```bash
python train.py --model resnet18 --dataset cifar10 --epochs 100 --batch_size 128 --lr 0.1
```

### 示例2: 训练ResNet50 on CIFAR-100

```bash
python train.py --model resnet50 --dataset cifar100 --epochs 100 --batch_size 128 --lr 0.1
```

### 示例3: 快速测试

```bash
python train.py --model resnet18 --dataset cifar10 --epochs 10 --batch_size 32
```

## 📝 命令行参数

- `--model`: 模型名称 (resnet18/34/50/101/152)
- `--dataset`: 数据集 (cifar10/cifar100)
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 128)
- `--lr`: 学习率 (默认: 0.1)

## 🎯 特性

✓ YOLO风格的输出格式  
✓ 自动保存best.pt和last.pt  
✓ 训练曲线可视化  
✓ 内存使用监控  
✓ 完整的日志记录  
✓ 所有输出统一到output文件夹  
✓ 模块化代码结构  

## 📂 数据划分

- 训练集: 80%
- 验证集: 20%
- 测试集: 10000个样本（固定）

可在 `config.py` 中修改 `TRAIN_VAL_SPLIT` 比例。

## 🔧 配置说明

所有配置都在 `config.py` 中：

- `TRAIN_CONFIG`: 训练超参数
- `DATA_CONFIG`: 数据集配置
- `MODEL_CONFIG`: 模型配置
- `AUGMENT_CONFIG`: 数据增强配置
- `TRAIN_VAL_SPLIT`: 数据划分比例
- `SAVE_CONFIG`: 保存配置

## 💡 预期结果

### CIFAR-10
- ResNet18: ~92% 准确率
- ResNet34: ~93% 准确率
- ResNet50: ~94% 准确率

### CIFAR-100
- ResNet18: ~70% 准确率
- ResNet50: ~75% 准确率
