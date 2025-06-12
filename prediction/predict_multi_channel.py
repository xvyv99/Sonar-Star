#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用多通道SE-ResNet50模型进行自闭症风险预测
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from multi_channel.model import MultiChannelSE_ResNet50
from multi_channel.extract_features import extract_all_features

def load_model(model_path, num_channels=5):
    """
    加载预训练的多通道SE-ResNet50模型
    
    Args:
        model_path: 模型文件路径
        num_channels: 输入通道数
        
    Returns:
        model: 加载的模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = MultiChannelSE_ResNet50(num_classes=2, num_channels=num_channels)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移至设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载，验证准确率: {checkpoint.get('val_acc', 'N/A')}, F1分数: {checkpoint.get('val_f1', 'N/A')}")
    
    return model, device

def prepare_input(feature_paths):
    """
    准备模型输入
    
    Args:
        feature_paths: 特征文件路径字典
        
    Returns:
        input_tensor: 模型输入张量
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # 加载并处理各种特征图像
    feature_images = []
    
    # MFCC特征
    if 'mfcc' in feature_paths and os.path.exists(feature_paths['mfcc']):
        mfcc_image = Image.open(feature_paths['mfcc']).convert('L')
        mfcc_tensor = transform(mfcc_image)
        feature_images.append(mfcc_tensor)
    else:
        feature_images.append(torch.zeros(1, 224, 224))
    
    # 声谱图特征
    if 'spectrogram' in feature_paths and os.path.exists(feature_paths['spectrogram']):
        spec_image = Image.open(feature_paths['spectrogram']).convert('L')
        spec_tensor = transform(spec_image)
        feature_images.append(spec_tensor)
    else:
        feature_images.append(torch.zeros(1, 224, 224))
    
    # 节奏特征
    if 'rhythm' in feature_paths and os.path.exists(feature_paths['rhythm']):
        rhythm_image = Image.open(feature_paths['rhythm']).convert('L')
        rhythm_tensor = transform(rhythm_image)
        feature_images.append(rhythm_tensor)
    else:
        feature_images.append(torch.zeros(1, 224, 224))
    
    # 共振峰特征
    if 'formant' in feature_paths and os.path.exists(feature_paths['formant']):
        formant_image = Image.open(feature_paths['formant']).convert('L')
        formant_tensor = transform(formant_image)
        feature_images.append(formant_tensor)
    else:
        feature_images.append(torch.zeros(1, 224, 224))
    
    # 质量特征
    if 'quality' in feature_paths and os.path.exists(feature_paths['quality']):
        quality_image = Image.open(feature_paths['quality']).convert('L')
        quality_tensor = transform(quality_image)
        feature_images.append(quality_tensor)
    else:
        feature_images.append(torch.zeros(1, 224, 224))
    
    # 将所有特征图像堆叠为一个多通道张量
    multi_channel_image = torch.cat(feature_images, dim=0)
    
    # 添加批次维度
    input_tensor = multi_channel_image.unsqueeze(0)
    
    return input_tensor

def predict_multi_channel(model, input_tensor, device):
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

def generate_report(audio_path, feature_paths, pred_class, pred_prob, output_dir):
    """
    生成预测报告
    
    Args:
        audio_path: 音频文件路径
        feature_paths: 特征文件路径字典
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
    
    # 写入报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"星语自闭症风险预测报告\n")
        f.write(f"====================\n\n")
        f.write(f"音频文件: {audio_path}\n")
        f.write(f"预测结果: {class_labels[pred_class]}\n")
        f.write(f"预测概率: {pred_prob:.2%}\n")
        f.write(f"风险等级: {risk_levels[pred_class]}\n\n")
        
        f.write(f"特征文件列表:\n")
        for feature_name, feature_path in feature_paths.items():
            f.write(f"- {feature_name}: {feature_path}\n")
    
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
    parser = argparse.ArgumentParser(description='使用多通道SE-ResNet50模型进行自闭症风险预测')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    parser.add_argument('--num_channels', type=int, default=5, help='输入通道数')
    args = parser.parse_args()
    
    # 提取所有特征
    feature_paths = extract_all_features(args.audio, args.output)
    
    # 加载模型
    model, device = load_model(args.model, args.num_channels)
    
    # 准备输入
    input_tensor = prepare_input(feature_paths)
    
    # 进行预测
    pred_class, pred_prob = predict_multi_channel(model, input_tensor, device)
    
    # 生成报告
    report_path = generate_report(args.audio, feature_paths, pred_class, pred_prob, args.output)
    
    # 输出预测结果
    class_labels = ['正常', '自闭症']
    print(f"预测结果: {class_labels[pred_class]}, 概率: {pred_prob:.2%}")
    print(f"报告已保存至: {report_path}")

if __name__ == '__main__':
    main() 