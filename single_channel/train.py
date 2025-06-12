#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单通道SE-ResNet50模型训练脚本
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms

from .model import SingleChannelSE_ResNet50, MFCCDataset, set_seed

def train_model(args):
    """
    训练单通道SE-ResNet50模型
    
    Args:
        args: 参数
    """
    print(f"\n开始训练单通道SE-ResNet50模型...")
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = MFCCDataset(
        data_dir=args.data_dir,
        labels_file=os.path.join(args.data_dir, 'train.csv'),
        transform=train_transform
    )
    
    val_dataset = MFCCDataset(
        data_dir=args.data_dir,
        labels_file=os.path.join(args.data_dir, 'val.csv'),
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # 创建模型
    model = SingleChannelSE_ResNet50(num_classes=2)
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=device))  # 给自闭症类别更高的权重
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.epochs // 3,  # 每T_0个epoch重启一次
        T_mult=2,  # 每次重启后，T_0变为T_0*T_mult
        eta_min=args.lr / 100  # 最小学习率
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    best_f1_score = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print(f"开始训练，总轮数: {args.epochs}")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        # 计算混淆矩阵和分类报告
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['Normal', 'ASD'])
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 打印结果
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"学习率: {current_lr:.6f}")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        print(f"分类报告:\n{report}")
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)
        
        # 保存检查点
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_f1': f1
        }, checkpoint_path)
        
        # 保存最佳模型 (基于准确率)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model_acc.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': f1
            }, best_model_path)
            print(f"最佳准确率模型已保存，验证准确率: {val_acc:.4f}")
        
        # 保存最佳F1分数模型
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_path = os.path.join(args.save_dir, 'best_model_f1.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': f1
            }, best_model_path)
            print(f"最佳F1分数模型已保存，F1分数: {f1:.4f}")
        
        # 复制最佳模型到标准位置
        if epoch == args.epochs - 1 or val_acc == best_val_acc:
            import shutil
            shutil.copy(best_model_path, os.path.join(args.save_dir, 'best_model.pth'))
    
    # 绘制训练历史
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失曲线')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.xlabel('轮数')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('准确率曲线')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='验证F1分数')
    plt.xlabel('轮数')
    plt.ylabel('F1分数')
    plt.legend()
    plt.title('F1分数曲线')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    print(f"训练完成。最佳验证准确率: {best_val_acc:.4f}, 最佳F1分数: {best_f1_score:.4f}")
    
    return best_val_acc, best_f1_score

def main():
    parser = argparse.ArgumentParser(description='单通道SE-ResNet50训练')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./model_training/single_channel', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    args = parser.parse_args()
    
    train_model(args)

if __name__ == '__main__':
    main() 