"""
日志记录模块 - YOLO风格的输出
"""

import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import torch


class Logger:
    """YOLO风格的日志记录器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_file = output_dir / "training.log"
        
        # 打开日志文件
        self.log_fp = open(self.log_file, 'w')
    
    def info(self, *args, **kwargs):
        """输出信息"""
        msg = ' '.join(str(arg) for arg in args)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        self.log_fp.write(log_msg + '\n')
        self.log_fp.flush()
    
    def __del__(self):
        if hasattr(self, 'log_fp'):
            self.log_fp.close()


class MetricsLogger:
    """指标记录器 - 用于每轮的训练指标"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics.json"
        self.metrics = []
        self.start_time = time.time()
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """记录一个epoch的指标"""
        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "learning_rate": float(lr),
            "memory_mb": self._get_memory_usage(),
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics.append(metrics)
        return metrics
    
    def _get_memory_usage(self):
        """获取内存使用量（MB）"""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def save(self):
        """保存所有指标到文件"""
        import json
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_epoch_summary(self, epoch_metrics):
        """打印epoch摘要 - YOLO风格"""
        epoch = epoch_metrics["epoch"]
        train_loss = epoch_metrics["train_loss"]
        train_acc = epoch_metrics["train_acc"]
        val_loss = epoch_metrics["val_loss"]
        val_acc = epoch_metrics["val_acc"]
        lr = epoch_metrics["learning_rate"]
        mem = epoch_metrics["memory_mb"]
        
        # 计算epoch时间
        elapsed = time.time() - self.start_time
        
        print(f"\nEpoch {epoch:3d}/100: "
              f"train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.2f}% "
              f"val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.2f}% "
              f"lr={lr:.6f} "
              f"mem={mem:.0f}MB "
              f"time={elapsed:.1f}s")

