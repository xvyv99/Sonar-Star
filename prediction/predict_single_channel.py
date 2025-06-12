#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用单通道SE-ResNet50模型进行自闭症风险预测
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import librosa
import librosa.display

from feature_extract.acoustic_feature import SpectralSubtraction

def extract_mfcc(audio_path, output_dir='./features'):
    """
    提取MFCC特征并保存为图像
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        
    Returns:
        mfcc_path: MFCC特征图像路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 文件名处理
    base_name = os.path.basename(audio_path).split('.')[0]
    
    print(f"提取MFCC特征: {audio_path}")
    
    # 1. 降噪处理
    audio_data, sr = librosa.load(audio_path, sr=None)
    ss = SpectralSubtraction(audio_data, sr)
    enhanced_audio = ss.BeroutiSpectralSubtraction()
    
    # 2. MFCC特征
    mfcc_path = os.path.join(output_dir, f"{base_name}_mfcc.png")
    plt.figure(figsize=(8, 8))
    mfccs = librosa.feature.mfcc(y=enhanced_audio, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.axis('off')  # 去掉坐标轴
    plt.tight_layout()
    plt.savefig(mfcc_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"MFCC特征已保存至: {mfcc_path}")
    
    return mfcc_path

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

class SingleChannelSE_ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(SingleChannelSE_ResNet50, self).__init__()
        # 加载预训练的ResNet50模型
        from torchvision.models import resnet50
        self.resnet = resnet50(pretrained=True)
        
        # 修改第一个卷积层以适应灰度图像
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 在每个残差块后添加SE块（仅添加属性，不注册hook）
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

def load_model(model_path):
    """
    加载预训练的单通道SE-ResNet50模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        model: 加载的模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SingleChannelSE_ResNet50(num_classes=2)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 使用strict=False忽略多余的参数
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # 将模型移至设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载，验证准确率: {checkpoint.get('val_acc', 'N/A')}, F1分数: {checkpoint.get('val_f1', 'N/A')}")
    
    return model, device

def prepare_input(mfcc_path):
    """
    准备模型输入
    
    Args:
        mfcc_path: MFCC特征图像路径
        
    Returns:
        input_tensor: 模型输入张量
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # 加载并处理MFCC特征图像
    mfcc_image = Image.open(mfcc_path).convert('L')
    mfcc_tensor = transform(mfcc_image)
    
    # 添加批次维度
    input_tensor = mfcc_tensor.unsqueeze(0)
    
    return input_tensor

def predict_single_channel(model, input_tensor, device):
    """
    使用模型进行预测
    
    Args:
        model: 预训练模型
        input_tensor: 输入张量
        device: 设备
        
    Returns:
        pred_class: 预测类别 (0: 正常, 1: 自闭症)
        pred_prob: 预测概率
    """
    # 将输入移至设备
    input_tensor = input_tensor.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # 获取预测类别和概率
    pred_class = torch.argmax(probabilities, dim=1).item()
    pred_prob = probabilities[0, pred_class].item()
    
    return pred_class, pred_prob

def generate_report(audio_path, mfcc_path, pred_class, pred_prob, output_dir):
    """
    生成预测报告
    
    Args:
        audio_path: 音频文件路径
        mfcc_path: MFCC特征图像路径
        pred_class: 预测类别
        pred_prob: 预测概率
        output_dir: 输出目录
        
    Returns:
        report_path: 报告文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成报告文件路径
    base_name = os.path.basename(audio_path).split('.')[0]
    report_path = os.path.join(output_dir, f"{base_name}_report.txt")
    
    # 类别标签
    class_labels = ['正常', '自闭症']
    risk_levels = ['低风险', '高风险']
    
    # 写入报告，使用UTF-8-SIG编码（带BOM）
    with open(report_path, 'w', encoding='utf-8-sig') as f:
        f.write(f"星语自闭症风险预测报告 (单通道模型)\n")
        f.write(f"====================\n\n")
        f.write(f"音频文件: {audio_path}\n")
        f.write(f"预测结果: {class_labels[pred_class]}\n")
        f.write(f"预测概率: {pred_prob:.2%}\n")
        f.write(f"风险等级: {risk_levels[pred_class]}\n\n")
        
        f.write(f"MFCC特征文件: {mfcc_path}\n")
    
    print(f"预测报告已保存至: {report_path}")
    
    # 生成可视化报告
    plt.figure(figsize=(10, 6))
    plt.bar(['正常', '自闭症'], [1-pred_prob if pred_class == 1 else pred_prob, pred_prob if pred_class == 1 else 1-pred_prob])
    plt.ylim(0, 1)
    plt.ylabel('概率')
    plt.title(f'自闭症风险预测结果: {class_labels[pred_class]} ({pred_prob:.2%})')
    
    # 保存图表
    chart_path = os.path.join(output_dir, f"{base_name}_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='使用单通道SE-ResNet50模型进行自闭症风险预测')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    args = parser.parse_args()
    
    # 提取MFCC特征
    mfcc_path = extract_mfcc(args.audio, args.output)
    
    # 加载模型
    model, device = load_model(args.model)
    
    # 准备输入
    input_tensor = prepare_input(mfcc_path)
    
    # 进行预测
    pred_class, pred_prob = predict_single_channel(model, input_tensor, device)
    
    # 生成报告
    report_path = generate_report(args.audio, mfcc_path, pred_class, pred_prob, args.output)
    
    # 输出预测结果
    class_labels = ['正常', '自闭症']
    print(f"预测结果: {class_labels[pred_class]}, 概率: {pred_prob:.2%}")
    print(f"报告已保存至: {report_path}")

if __name__ == '__main__':
    main() 