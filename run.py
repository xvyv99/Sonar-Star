#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
星语自闭症早期筛查系统主入口
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
    parser = argparse.ArgumentParser(description='星语自闭症早期筛查系统')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='multi', help='运行模式: single(单通道)或multi(多通道)')
    parser.add_argument('--audio_dir', type=str, default='./data', help='音频文件目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--test_audio', type=str, default=None, help='用于测试的音频文件（可选）')
    parser.add_argument('--skip_training', action='store_true', help='跳过训练步骤')
    parser.add_argument('--skip_data_prep', action='store_true', help='跳过数据准备步骤')
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n星语自闭症早期筛查系统")
    print(f"运行模式: {'单通道' if args.mode == 'single' else '多通道'}")
    print(f"开始运行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 根据模式选择运行脚本
    if args.mode == 'single':
        run_command(
            f"python -m single_channel.run --audio_dir {args.audio_dir} --output_dir {args.output_dir} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} "
            f"{'--test_audio ' + args.test_audio if args.test_audio else ''} "
            f"{'--skip_training' if args.skip_training else ''} "
            f"{'--skip_data_prep' if args.skip_data_prep else ''}",
            "运行单通道模型流程"
        )
    else:
        run_command(
            f"python -m multi_channel.run --audio_dir {args.audio_dir} --output_dir {args.output_dir} "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} "
            f"{'--test_audio ' + args.test_audio if args.test_audio else ''} "
            f"{'--skip_training' if args.skip_training else ''} "
            f"{'--skip_data_prep' if args.skip_data_prep else ''}",
            "运行多通道模型流程"
        )
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n结束运行时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("\n星语自闭症早期筛查系统运行完成！")

if __name__ == '__main__':
    main() 