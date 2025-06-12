#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练多通道SE-ResNet50模型
用于自闭症早期筛查
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from multi_channel.model import MultiChannelSE_ResNet50, MultiFeatureDataset, train, validate

def main():
    parser = argparse.ArgumentParser(description='训练多通道SE-ResNet50模型')
    parser.add_argument('--data_dir', type=str, default='./results', help='数据目录')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='训练数据CSV文件')
    parser.add_argument('--val_csv', type=str, default='val.csv', help='验证数据CSV文件')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--save_dir', type=str, default='./model_training/model_output', help='模型保存目录')
    parser.add_argument('--num_channels', type=int, default=5, help='输入通道数')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = MultiFeatureDataset(
        base_dir=args.data_dir,
        labels_file=os.path.join(args.data_dir, args.train_csv),
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = MultiFeatureDataset(
        base_dir=args.data_dir,
        labels_file=os.path.join(args.data_dir, args.val_csv),
        transform=val_transform,
        mode='val'
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
    model = MultiChannelSE_ResNet50(num_classes=2, num_channels=args.num_channels)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 训练循环
    best_val_acc = 0.0
    best_f1_score = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, cm, report, f1 = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        print(f"分类报告:\n{report}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型 (基于准确率)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': f1
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"最佳模型已保存，验证准确率: {val_acc:.4f}")
    
    print(f"训练完成。最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main() 