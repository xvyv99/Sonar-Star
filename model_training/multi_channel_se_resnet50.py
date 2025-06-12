#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多通道SE-ResNet50模型
用于从多种声学特征图像预测自闭症谱系障碍的概率
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torchvision.transforms.functional as TF
from PIL import Image

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义SE Block (Squeeze-and-Excitation Block)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义多通道SE-ResNet50模型
class MultiChannelSE_ResNet50(nn.Module):
    def __init__(self, num_classes=2, num_channels=5):
        super(MultiChannelSE_ResNet50, self).__init__()
        # 加载预训练的ResNet50模型
        self.resnet = models.resnet50(pretrained=True)
        
        # 修改第一个卷积层以适应多通道图像
        # 5个通道：MFCC、声谱图、短时能量、过零率、共振峰
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 在每个残差块后添加SE块
        for name, module in self.resnet.named_children():
            if name == 'layer1' or name == 'layer2' or name == 'layer3' or name == 'layer4':
                for bottleneck in module:
                    bottleneck.se_block = SEBlock(bottleneck.conv3.out_channels)
                    
        # 修改最后的全连接层以适应我们的分类任务
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 使用正常的ResNet前向传播
        x = self.resnet(x)
        return x

# 自定义多通道数据集类
class MultiFeatureDataset(Dataset):
    def __init__(self, base_dir, labels_file=None, transform=None, mode='train'):
        """
        多特征图像数据集
        Args:
            base_dir (str): 基础目录，包含各种特征子目录
            labels_file (str): 标签文件路径，CSV格式，包含image_name和label列
            transform (callable, optional): 数据增强和预处理
            mode (str): 'train', 'val', 或 'test'
        """
        self.base_dir = base_dir
        self.transform = transform
        self.mode = mode
        
        # 特征子目录
        self.feature_dirs = {
            'mfcc': os.path.join(base_dir, 'mfcc'),
            'spectrogram': os.path.join(base_dir, 'spectrogram'),
            'rhythm': os.path.join(base_dir, 'rhythm'),
            'formant': os.path.join(base_dir, 'formant'),
            'quality': os.path.join(base_dir, 'quality')
        }
        
        if mode == 'test' and labels_file is None:
            # 测试模式且无标签文件，直接使用目录中的所有图像
            # 获取MFCC目录中的所有文件名作为基准
            self.base_names = [f.split('_mfcc.png')[0] for f in os.listdir(self.feature_dirs['mfcc']) 
                               if f.endswith('.png')]
            self.labels = None
        else:
            # 训练或验证模式，需要标签文件
            df = pd.read_csv(labels_file)
            self.base_names = [f.split('.')[0] for f in df['image_name']]
            self.labels = df['label'].values  # 0: 正常, 1: 自闭症

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base_name = self.base_names[idx]
        
        # 加载各种特征图像
        feature_images = []
        
        # MFCC特征
        mfcc_path = os.path.join(self.feature_dirs['mfcc'], f"{base_name}_mfcc.png")
        if os.path.exists(mfcc_path):
            mfcc_image = Image.open(mfcc_path).convert('L')  # 转换为灰度图
            if self.transform:
                mfcc_image = self.transform(mfcc_image)
            feature_images.append(mfcc_image)
        else:
            # 如果文件不存在，创建空白图像
            feature_images.append(torch.zeros(1, 224, 224))
        
        # 声谱图特征
        spec_path = os.path.join(self.feature_dirs['spectrogram'], f"{base_name}_spectrogram.png")
        if os.path.exists(spec_path):
            spec_image = Image.open(spec_path).convert('L')
            if self.transform:
                spec_image = self.transform(spec_image)
            feature_images.append(spec_image)
        else:
            feature_images.append(torch.zeros(1, 224, 224))
        
        # 节奏特征 (短时能量和过零率)
        rhythm_path = os.path.join(self.feature_dirs['rhythm'], f"{base_name}_rhythm.png")
        if os.path.exists(rhythm_path):
            rhythm_image = Image.open(rhythm_path).convert('L')
            if self.transform:
                rhythm_image = self.transform(rhythm_image)
            feature_images.append(rhythm_image)
        else:
            feature_images.append(torch.zeros(1, 224, 224))
        
        # 共振峰特征
        formant_path = os.path.join(self.feature_dirs['formant'], f"{base_name}_formant.png")
        if os.path.exists(formant_path):
            formant_image = Image.open(formant_path).convert('L')
            if self.transform:
                formant_image = self.transform(formant_image)
            feature_images.append(formant_image)
        else:
            feature_images.append(torch.zeros(1, 224, 224))
        
        # 质量特征
        quality_path = os.path.join(self.feature_dirs['quality'], f"{base_name}_quality.png")
        if os.path.exists(quality_path):
            quality_image = Image.open(quality_path).convert('L')
            if self.transform:
                quality_image = self.transform(quality_image)
            feature_images.append(quality_image)
        else:
            feature_images.append(torch.zeros(1, 224, 224))
        
        # 将所有特征图像堆叠为一个多通道张量
        multi_channel_image = torch.cat(feature_images, dim=0)
        
        if self.mode == 'test' and self.labels is None:
            return multi_channel_image, base_name
        else:
            label = self.labels[idx]
            return multi_channel_image, label

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
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
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 计算混淆矩阵和分类报告
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'ASD'])
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, cm, report, f1

# 主函数
def main():
    parser = argparse.ArgumentParser(description='多通道SE-ResNet50训练')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='训练数据CSV文件')
    parser.add_argument('--val_csv', type=str, default='val.csv', help='验证数据CSV文件')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--num_channels', type=int, default=5, help='输入通道数')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
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
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    best_f1_score = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
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
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)
        
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
            }, os.path.join(args.save_dir, 'best_acc_model.pth'))
            print(f"最佳准确率模型已保存，验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型 (基于F1分数)
        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': f1
            }, os.path.join(args.save_dir, 'best_f1_model.pth'))
            print(f"最佳F1分数模型已保存，F1分数: {f1:.4f}")
        
        # 每个epoch都保存一个checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_f1': f1
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
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

if __name__ == '__main__':
    main() 