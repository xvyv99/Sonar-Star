#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一键运行多通道模型的完整流程
包括特征提取、数据准备、模型训练和预测
"""

import os
import argparse
import subprocess
import time
from datetime import datetime

def run_command(command, description=None):
    """
    运行命令并打印输出
    
    Args:
        command: 要运行的命令
        description: 命令描述
    """
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}\n")
    
    print(f"执行命令: {command}")
    start_time = time.time()
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        print(f"命令成功完成，耗时: {end_time - start_time:.2f} 秒")
    else:
        print(f"命令执行失败，退出代码: {process.returncode}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description='一键运行多通道模型的完整流程')
    parser.add_argument('--audio_dir', type=str, default='./data', help='音频文件目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--test_audio', type=str, default=None, help='用于测试的音频文件（可选）')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip_data_prep', action='store_true', help='跳过数据准备步骤')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join('model_training', 'model_output')
    os.makedirs(model_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 数据准备
    if not args.skip_data_prep:
        run_command(
            f"python -m multi_channel.prepare_data --audio_dir {args.audio_dir} --output_dir {args.output_dir}",
            "步骤1: 批量处理音频文件并生成多通道数据集"
        )
    else:
        print("\n跳过数据准备步骤...")
    
    # 步骤2: 模型训练
    if not args.skip_training:
        run_command(
            f"python -m multi_channel.train --data_dir {args.output_dir} --epochs {args.epochs} "
            f"--batch_size {args.batch_size} --lr {args.lr} --save_dir {model_dir}",
            "步骤2: 训练多通道SE-ResNet50模型"
        )
    else:
        print("\n跳过模型训练步骤...")
    
    # 步骤3: 模型预测（如果提供了测试音频）
    if args.test_audio:
        model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            run_command(
                f"python -m prediction.predict_multi_channel --audio {args.test_audio} --model {model_path} --output {args.output_dir}",
                "步骤3: 使用多通道模型进行自闭症风险预测"
            )
        else:
            print(f"\n警告: 模型文件 {model_path} 不存在，跳过预测步骤...")
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n结束运行时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("\n流程已完成！")

if __name__ == '__main__':
    main() 