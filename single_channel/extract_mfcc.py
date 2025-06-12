#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取MFCC特征并保存为图像
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

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

def main():
    parser = argparse.ArgumentParser(description='提取MFCC特征')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--output', type=str, default='./features', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 提取MFCC特征
    mfcc_path = extract_mfcc(args.audio, args.output)
    
    print(f"MFCC特征已保存至: {mfcc_path}")

if __name__ == '__main__':
    main() 