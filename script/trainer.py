"""
训练器模块 - 包含训练逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
import os

from config import TRAIN_CONFIG, OUTPUT_DIR, SAVE_CONFIG
from logger import Logger, MetricsLogger


class Trainer:
    """训练器类"""
    
    def __init__(self, model, device, train_loader, val_loader, test_loader, 
                 model_name, dataset, output_dir):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            momentum=TRAIN_CONFIG["momentum"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        # 学习率调度器
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=TRAIN_CONFIG["milestones"],
            gamma=TRAIN_CONFIG["gamma"]
        )
        
        # 日志
        self.logger = Logger(output_dir)
        self.metrics_logger = MetricsLogger(output_dir)
        
        # 最佳准确率
        self.best_val_acc = 0.0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """验证/测试"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        
        # 保存最新
        torch.save(checkpoint, self.output_dir / 'last.pt')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            self.logger.info(f"Saved best model: best.pt (val_acc={val_acc:.2f}%)")
        
        # 定期保存
        if epoch % SAVE_CONFIG["save_freq"] == 0:
            torch.save(checkpoint, self.output_dir / f'epoch_{epoch}.pt')
    
    def train(self):
        """主训练循环"""
        num_epochs = TRAIN_CONFIG["num_epochs"]
        
        self.logger.info("="*60)
        self.logger.info("开始训练")
        self.logger.info("="*60)
        self.logger.info(f"模型: {self.model_name}")
        self.logger.info(f"数据集: {self.dataset}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Batch Size: {TRAIN_CONFIG['batch_size']}")
        self.logger.info(f"Learning Rate: {TRAIN_CONFIG['learning_rate']}")
        self.logger.info("="*60)
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(self.val_loader)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # 记录指标
            epoch_metrics = self.metrics_logger.log_epoch(
                epoch, train_loss, train_acc, val_loss, val_acc, current_lr
            )
            
            # 打印摘要
            self.metrics_logger.print_epoch_summary(epoch_metrics)
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        # 在测试集上评估
        self.logger.info("\n" + "="*60)
        self.logger.info("在测试集上评估...")
        self.logger.info("="*60)
        
        test_loss, test_acc = self.validate(self.test_loader)
        self.logger.info(f"测试集准确率: {test_acc:.2f}%")
        self.logger.info(f"测试集损失: {test_loss:.4f}")
        
        # 保存指标
        self.metrics_logger.save()
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        self.logger.info("="*60)

