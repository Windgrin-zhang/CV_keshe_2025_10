# 训练系统使用指南

## ✅ 重构完成

已按你的要求完成所有修改：

### 1. ✅ 可视化输出
- 生成训练曲线图（准确率和损失）
- 学习率变化曲线
- 内存使用监控
- YOLO风格的输出格式

### 2. ✅ 超参数配置
- 已在config.py中设置推荐的超参数
- 学习率: 0.1 (CIFAR标准)
- 批量大小: 128
- 权重衰减: 5e-4
- 学习率衰减: milestones=[60, 80], gamma=0.1

### 3. ✅ Bash输出
每轮输出包括：
```
Epoch   1/100: train_loss=2.5362 train_acc=19.61% val_loss=1.9650 val_acc=23.62% lr=0.100000 mem=1162MB time=7.1s
```
- 训练/验证准确率
- 训练/验证损失
- 学习率
- 内存占用
- 训练时间

### 4. ✅ 输出best.pt
- 自动保存best.pt（最佳模型）
- 自动保存last.pt（最新模型）
- 每10个epoch保存一次检查点

### 5. ✅ 函数封装
train.py只包含：
- 主函数
- 超参设置
- 模型选择
- 输入输出路径

### 6. ✅ 数据划分配置
在config.py中：
```python
TRAIN_VAL_SPLIT = {
    "train_ratio": 0.8,  # 80%训练
    "val_ratio": 0.2,     # 20%验证
}
```

### 7. ✅ 模块化重构
```python
script/
├── train.py       # 主函数（精简）
├── config.py      # 配置
├── data_loader.py # 数据加载
├── model.py       # 模型构建
├── trainer.py     # 训练逻辑
├── logger.py      # 日志（YOLO风格）
├── utils.py       # 可视化
└── evaluate.py    # 评估
```

### 8. ✅ 输出文件夹统一
- 所有输出在 `output/` 目录
- 格式：`output/{model}_{dataset}/`
- 包含：best.pt, last.pt, 训练日志, 指标JSON, 可视化图表

## 🚀 使用方法

### 快速开始
```bash
cd script
python train.py --epochs 10 --batch_size 64
```

### 完整训练
```bash
python train.py --model resnet50 --dataset cifar10 --epochs 100 --batch_size 128 --lr 0.1
```

### 查看结果
```bash
ls output/resnet18_cifar10/
# 查看训练日志
cat output/resnet18_cifar10/training.log
# 查看指标
cat output/resnet18_cifar10/metrics.json
```

## 📊 输出文件

每次训练会在output目录生成：

1. **best.pt** - 最佳模型权重
2. **last.pt** - 最新模型权重
3. **training.log** - 完整训练日志
4. **metrics.json** - 每轮指标（JSON格式）
5. **training_curves.png** - 准确率和损失曲线
6. **lr_curve.png** - 学习率曲线
7. **memory_usage.png** - 内存使用曲线

## 🎯 文件结构

### config.py
- 配置所有超参数
- 修改模型和数据选择
- 调整训练/验证比例

### train.py（主函数）
```python
def main():
    args = parse_args()  # 解析参数
    device = torch.device(...)  # 设备
    train_loader, val_loader, test_loader = load_cifar_dataset(...)  # 数据
    model = build_model(...)  # 模型
    trainer = Trainer(...)  # 训练器
    trainer.train()  # 训练
```

### 其他模块
- model.py: 模型构建
- trainer.py: 训练循环
- logger.py: YOLO风格日志
- utils.py: 可视化
- data_loader.py: 数据加载

## 💡 关键改进

1. **统一输出目录**：全部在output文件夹
2. **YOLO风格日志**：简洁清晰的训练输出
3. **完整可视化**：曲线图、学习率、内存
4. **模块化设计**：每个功能独立文件
5. **最佳模型保存**：自动保存best.pt
6. **配置化管理**：所有参数在config.py
7. **数据比例可调**：train/val比例可配置

## 🎓 使用建议

1. **首次运行**：使用少量epoch测试
2. **完整训练**：使用100 epochs
3. **查看结果**：检查best.pt和可视化图表
4. **调整参数**：在config.py中修改超参数

## 📈 预期性能

- CIFAR-10: 85-95% 准确率
- CIFAR-100: 70-80% 准确率

训练时长（100 epochs）：
- ResNet18: ~2-3小时
- ResNet50: ~5-6小时

