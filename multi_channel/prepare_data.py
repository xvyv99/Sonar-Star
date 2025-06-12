#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量处理音频文件并生成用于多通道模型的数据集
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from multi_channel.extract_features import extract_all_features

def process_audio_files(audio_dir, output_dir, label_file=None):
    """
    批量处理音频文件并提取特征
    
    Args:
        audio_dir: 音频文件目录
        output_dir: 输出目录
        label_file: 标签文件路径 (可选)
        
    Returns:
        processed_files: 处理后的文件信息列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 加载标签信息 (如果有)
    labels = {}
    if label_file and os.path.exists(label_file):
        label_df = pd.read_csv(label_file)
        for _, row in label_df.iterrows():
            labels[row['file_name']] = row['label']
        print(f"已加载 {len(labels)} 个标签")
    
    # 处理每个音频文件
    processed_files = []
    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        audio_path = os.path.join(audio_dir, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        
        # 提取所有特征
        feature_paths = extract_all_features(audio_path, output_dir)
        
        # 确定标签
        label = -1  # 默认未知
        if audio_file in labels:
            label = labels[audio_file]
        elif "ASD" in audio_file or "asd" in audio_file:
            label = 1  # 自闭症
        elif "Normal" in audio_file or "normal" in audio_file:
            label = 0  # 正常
        
        # 记录处理信息
        processed_files.append({
            'file_name': audio_file,
            'base_name': base_name,
            'label': label,
            'mfcc_path': feature_paths.get('mfcc', ''),
            'spectrogram_path': feature_paths.get('spectrogram', ''),
            'rhythm_path': feature_paths.get('rhythm', ''),
            'formant_path': feature_paths.get('formant', ''),
            'quality_path': feature_paths.get('quality', '')
        })
    
    return processed_files

def create_dataset_csv(processed_files, output_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    创建训练、验证和测试数据集的CSV文件
    
    Args:
        processed_files: 处理后的文件信息列表
        output_dir: 输出目录
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    """
    # 过滤掉没有标签的文件
    labeled_files = [f for f in processed_files if f['label'] >= 0]
    unlabeled_files = [f for f in processed_files if f['label'] < 0]
    
    print(f"标记文件数: {len(labeled_files)}, 未标记文件数: {len(unlabeled_files)}")
    
    if len(labeled_files) == 0:
        print("警告: 没有找到带标签的文件，无法创建训练/验证/测试集")
        return
    
    # 创建数据帧
    df = pd.DataFrame(labeled_files)
    
    # 分割训练集和测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_val_df['label']
    )
    
    # 创建图像名称列
    train_df['image_name'] = train_df['base_name'] + '.png'
    val_df['image_name'] = val_df['base_name'] + '.png'
    test_df['image_name'] = test_df['base_name'] + '.png'
    
    # 保存CSV文件
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    
    train_df[['image_name', 'label']].to_csv(train_csv_path, index=False)
    val_df[['image_name', 'label']].to_csv(val_csv_path, index=False)
    test_df[['image_name', 'label']].to_csv(test_csv_path, index=False)
    
    # 保存所有处理文件的信息
    all_files_csv_path = os.path.join(output_dir, 'all_files.csv')
    pd.DataFrame(processed_files).to_csv(all_files_csv_path, index=False)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"CSV文件已保存至: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='批量处理音频文件并生成多通道数据集')
    parser.add_argument('--audio_dir', type=str, required=True, help='音频文件目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--label_file', type=str, default=None, help='标签文件路径 (CSV格式，包含file_name和label列)')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 批量处理音频文件
    processed_files = process_audio_files(args.audio_dir, args.output_dir, args.label_file)
    
    # 创建数据集CSV文件
    create_dataset_csv(
        processed_files, args.output_dir, 
        test_size=args.test_size, val_size=args.val_size, random_state=args.random_state
    )

if __name__ == '__main__':
    main() 