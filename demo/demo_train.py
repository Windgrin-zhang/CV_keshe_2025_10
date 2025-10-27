"""
完整训练演示 - 运行一个epoch确保所有功能正常
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))

print("="*70)
print("完整训练演示 - 1个epoch快速测试")
print("="*70)

# 修改配置为少量epoch
from config import TRAIN_CONFIG
TRAIN_CONFIG["num_epochs"] = 1  # 只训练1个epoch
TRAIN_CONFIG["batch_size"] = 64  # 使用较小的batch
TRAIN_CONFIG["step_size"] = 1  # 确保不触发学习率衰减

# 暂时禁用TensorBoard（可能没有安装）
from config import LOG_CONFIG
LOG_CONFIG["use_tensorboard"] = False

# 暂时禁用tqdm（确保兼容性）
LOG_CONFIG["use_tqdm"] = False

print("\n配置:")
print(f"  模型: resnet18")
print(f"  数据集: cifar10")
print(f"  Epochs: {TRAIN_CONFIG['num_epochs']}")
print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
print(f"  学习率: {TRAIN_CONFIG['learning_rate']}\n")

# 导入并运行训练
try:
    from train import Trainer
    
    trainer = Trainer(
        model_name="resnet18",
        dataset="cifar10",
        device="cuda"
    )
    
    history = trainer.train()
    
    print("\n" + "="*70)
    print("✓ 完整训练演示成功！")
    print("="*70)
    print(f"\n训练结果:")
    print(f"  训练准确率: {history['train_acc'][-1]:.2f}%")
    print(f"  验证准确率: {history['val_acc'][-1]:.2f}%")
    print(f"\n现在可以运行完整的训练:")
    print(f"  cd script && python train.py --model resnet18 --dataset cifar10 --epochs 10")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()

