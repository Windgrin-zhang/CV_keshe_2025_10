"""
结果分析脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_results():
    """分析训练结果"""
    
    print("="*70)
    print("训练结果分析")
    print("="*70)
    
    # 读取历史数据
    history_path = Path("/home/alex/VScode/课程设计/results/resnet18_cifar10_history.json")
    
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print("\n【训练统计】")
        print(f"训练轮数: {len(history['train_acc'])} epochs")
        
        if len(history['train_acc']) > 0:
            print(f"\n【最终结果】")
            print(f"  训练准确率: {history['train_acc'][-1]:.2f}%")
            print(f"  验证准确率: {history['val_acc'][-1]:.2f}%")
            print(f"  训练损失: {history['train_loss'][-1]:.4f}")
            print(f"  验证损失: {history['val_loss'][-1]:.4f}")
            
            print(f"\n【最佳结果】")
            best_idx = np.argmax(history['val_acc'])
            print(f"  最佳验证准确率: {history['val_acc'][best_idx]:.2f}% (Epoch {best_idx+1})")
            print(f"  对应训练准确率: {history['train_acc'][best_idx]:.2f}%")
            
            print(f"\n【性能趋势分析】")
            # 检查过拟合
            final_gap = history['val_acc'][-1] - history['train_acc'][-1]
            if final_gap > 5:
                print(f"  ⚠️  可能存在过拟合 (验证-训练差距: {final_gap:.2f}%)")
            elif final_gap < -5:
                print(f"  ⚠️  训练不足 (训练准确率低于验证准确率)")
            else:
                print(f"  ✓  模型训练良好 (验证-训练差距: {final_gap:.2f}%)")
            
            # 检查收敛
            if len(history['val_acc']) >= 10:
                recent_trend = np.mean(history['val_acc'][-3:]) - np.mean(history['val_acc'][-10:-3])
                if recent_trend < 0.5:
                    print(f"  ✓  模型已收敛")
                else:
                    print(f"  ↝  模型仍在学习中")
        else:
            print("  无训练数据")
    else:
        print("\n❌ 未找到训练历史文件")
        print(f"   路径: {history_path}")
    
    print("\n" + "="*70)
    print("【结果解读】")
    print("="*70)
    print("\n从刚才的测试结果来看：")
    print("\n1. 【基础设置】")
    print("   - 模型: ResNet18 (轻量级, 适合CIFAR-10)")
    print("   - 数据集: CIFAR-10 (50,000训练样本, 10类)")
    print("   - 批次大小: 32 (较小，适合快速测试)")
    print("   - 训练轮数: 1 epoch (仅测试用)")
    
    print("\n2. 【训练结果】")
    print("   - 训练准确率: 57.88%")
    print("   - 验证准确率: 71.68%")
    print("   - 测试准确率: 72.77%")
    
    print("\n3. 【结果分析】")
    print("   ✓ 模型加载成功")
    print("   ✓ 数据加载和预处理正常")
    print("   ✓ 训练流程完整运行")
    print("   ✓ 只训练1个epoch就能达到72%准确率，说明：")
    print("     - 预训练权重有效（ImageNet预训练）")
    print("     - 迁移学习起作用")
    print("     - 模型架构适合CIFAR-10")
    
    print("\n   ⚠️  验证准确率(71.68%)高于训练准确率(57.88%)，可能原因：")
    print("     - 验证集较小（10,000样本 vs 40,000训练样本）")
    print("     - 数据增强使训练更困难但验证更容易")
    print("     - 模型在验证集上偶然表现好")
    print("     - 需要更多epoch才能看到真实的训练/验证趋势")
    
    print("\n4. 【改进建议】")
    print("   对于完整训练（100 epochs），预期：")
    print("   - 训练准确率: 85-95%")
    print("   - 验证准确率: 80-90%")
    print("   - 测试准确率: 80-90%")
    print("   - 建议batch_size: 128-256")
    print("   - 建议学习率: 0.001-0.01")
    
    print("\n5. 【下一步操作】")
    print("   运行完整训练：")
    print("   cd /home/alex/VScode/课程设计/script")
    print("   python train.py --model resnet18 --dataset cifar10 --epochs 100")
    print("\n   或使用ResNet50获得更高准确率：")
    print("   python train.py --model resnet50 --dataset cifar10 --epochs 100")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_training_results()

