#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取所有声学特征并保存为图像
包括：MFCC、声谱图、短时能量和过零率、共振峰
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from feature_extract.acoustic_feature import Spectrogram, SpectralSubtraction, VAD, RhythmFeatures, SpectrumFeatures, QualityFeatures

def extract_all_features(audio_path, output_dir='./features'):
    """
    提取所有声学特征并保存为图像
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        
    Returns:
        feature_paths: 特征文件路径字典
    """
    # 创建特征子目录
    feature_dirs = {
        'mfcc': os.path.join(output_dir, 'mfcc'),
        'spectrogram': os.path.join(output_dir, 'spectrogram'),
        'rhythm': os.path.join(output_dir, 'rhythm'),
        'formant': os.path.join(output_dir, 'formant'),
        'quality': os.path.join(output_dir, 'quality')
    }
    
    for dir_path in feature_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 文件名处理
    base_name = os.path.basename(audio_path).split('.')[0]
    
    print(f"提取音频特征: {audio_path}")
    
    # 创建特征路径字典
    feature_paths = {}
    
    # 1. 降噪处理
    audio_data, sr = librosa.load(audio_path, sr=None)
    ss = SpectralSubtraction(audio_data, sr)
    enhanced_audio = ss.BeroutiSpectralSubtraction()
    
    # 2. MFCC特征
    mfcc_path = os.path.join(feature_dirs['mfcc'], f"{base_name}_mfcc.png")
    plt.figure(figsize=(8, 8))
    mfccs = librosa.feature.mfcc(y=enhanced_audio, sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.axis('off')  # 去掉坐标轴
    plt.tight_layout()
    plt.savefig(mfcc_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    feature_paths['mfcc'] = mfcc_path
    print(f"MFCC特征已保存至: {mfcc_path}")
    
    # 3. 声谱图特征
    spec_path = os.path.join(feature_dirs['spectrogram'], f"{base_name}_spectrogram.png")
    sg = Spectrogram(audio_path)
    sg.plot(spec_path, show=False)
    feature_paths['spectrogram'] = spec_path
    print(f"声谱图特征已保存至: {spec_path}")
    
    # 4. 短时能量和过零率特征
    rhythm_path = os.path.join(feature_dirs['rhythm'], f"{base_name}_rhythm.png")
    rf = RhythmFeatures(audio_path)
    rf.plot(rhythm_path, show=False)
    feature_paths['rhythm'] = rhythm_path
    print(f"短时能量和过零率特征已保存至: {rhythm_path}")
    
    # 5. 共振峰特征
    formant_path = os.path.join(feature_dirs['formant'], f"{base_name}_formant.png")
    quality_path = os.path.join(feature_dirs['quality'], f"{base_name}_quality.png")
    qf = QualityFeatures(audio_path)
    qf.plot(formant_path, quality_path, show=False)
    feature_paths['formant'] = formant_path
    feature_paths['quality'] = quality_path
    print(f"共振峰特征已保存至: {formant_path}")
    print(f"质量特征已保存至: {quality_path}")
    
    print(f"所有音频特征已保存至: {output_dir}")
    
    return feature_paths

def main():
    parser = argparse.ArgumentParser(description='提取所有声学特征')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 提取所有特征
    feature_paths = extract_all_features(args.audio, args.output)
    
    # 生成特征报告
    report_path = os.path.join(args.output, 'feature_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"音频文件: {args.audio}\n")
        f.write("\n特征文件列表:\n")
        for feature_name, feature_path in feature_paths.items():
            f.write(f"{feature_name}: {feature_path}\n")
    
    print(f"特征报告已保存至: {report_path}")

if __name__ == '__main__':
    main() 