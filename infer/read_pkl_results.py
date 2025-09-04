#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取和打印 deepconf_online_generation.py 保存的 pkl 结果文件
"""

import pickle
import os
import glob
import json
from datetime import datetime
import numpy as np


def format_timestamp(timestamp_str):
    """格式化时间戳字符串"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def print_trace_summary(traces, phase_name):
    """打印trace汇总信息"""
    if not traces:
        print(f"  {phase_name}: 无traces")
        return
    
    print(f"  {phase_name}: {len(traces)}个traces")
    
    # 统计final_score分布
    final_scores = [t.get('final_score', 0) for t in traces]
    if final_scores:
        print(f"    Final Score - 平均: {np.mean(final_scores):.4f}, 标准差: {np.std(final_scores):.4f}")
        print(f"    Final Score - 最小: {np.min(final_scores):.4f}, 最大: {np.max(final_scores):.4f}")
    
    # 统计token数量
    token_counts = [t.get('token_count', 0) for t in traces]
    if token_counts:
        print(f"    Token数量 - 总计: {sum(token_counts)}, 平均: {np.mean(token_counts):.1f}")
    
    # 统计置信度
    min_confs = [t.get('min_conf', 0) for t in traces]
    if min_confs:
        print(f"    最小置信度 - 平均: {np.mean(min_confs):.4f}, 标准差: {np.std(min_confs):.4f}")


def print_voting_info(voting_data):
    """打印投票信息"""
    if not voting_data:
        print("  无投票数据")
        return
    
    print(f"  投票答案数量: {voting_data.get('voting_traces_count', 0)}")
    print(f"  投票总token数: {voting_data.get('voting_total_tokens', 0)}")
    print(f"  最终投票答案: {voting_data.get('voted_answer', 'None')}")
    print(f"  Net得分: {voting_data.get('net_score', 0):.4f}")
    print(f"  Dismantle得分: {voting_data.get('dismantle_score', 0):.4f}")
    print(f"  最终得分: {voting_data.get('final_score', 0):.4f}")
    print(f"  是否正确(阈值0.8): {'✅' if voting_data.get('is_voted_correct', False) else '❌'}")


def print_statistics(stats):
    """打印统计信息"""
    print("\n📊 统计信息:")
    
    # 按阶段打印统计
    phases = ['warmup', 'final', 'overall']
    for phase in phases:
        phase_stats = {k: v for k, v in stats.items() if k.startswith(phase)}
        if phase_stats:
            print(f"\n  {phase.upper()}阶段:")
            for key, value in phase_stats.items():
                clean_key = key.replace(f"{phase}_", "")
                if isinstance(value, float):
                    print(f"    {clean_key}: {value:.4f}")
                else:
                    print(f"    {clean_key}: {value}")
    
    # 性能指标
    performance_keys = ['total_time', 'warmup_time', 'final_time', 'tokens_per_second', 'traces_per_second']
    performance_stats = {k: v for k, v in stats.items() if k in performance_keys}
    if performance_stats:
        print(f"\n  性能指标:")
        for key, value in performance_stats.items():
            if 'time' in key:
                print(f"    {key}: {value:.2f}秒")
            else:
                print(f"    {key}: {value:.2f}")


def read_pkl_file(pkl_path):
    """读取单个pkl文件"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def print_pkl_summary(data, filename):
    """打印pkl文件内容摘要"""
    print("=" * 80)
    print(f"📁 文件: {filename}")
    print("=" * 80)
    
    # 基本信息
    print("\n🔍 基本信息:")
    print(f"  问题ID: {data.get('question_id', 'N/A')}")
    print(f"  运行ID: {data.get('run_id', 'N/A')}")
    print(f"  时间戳: {format_timestamp(data.get('timestamp', 'N/A'))}")
    print(f"  置信度阈值: {data.get('conf_bar', 0):.4f}")
    
    # 问题和真实答案
    print(f"\n❓ 问题: {data.get('question', 'N/A')}")
    print(f"✅ 真实答案: {data.get('ground_truth', 'N/A')}")
    
    # 配置信息
    config = data.get('config', {})
    if config:
        print(f"\n⚙️ 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Traces信息
    print(f"\n🎯 Traces信息:")
    warmup_traces = data.get('warmup_traces', [])
    final_traces = data.get('final_traces', [])
    
    print_trace_summary(warmup_traces, "Warmup")
    print_trace_summary(final_traces, "Final")
    
    # 投票信息
    voting_data = data.get('voting', {})
    if voting_data:
        print(f"\n🗳️ 投票结果:")
        print_voting_info(voting_data)
    
    # 统计信息
    statistics = data.get('statistics', {})
    if statistics:
        print_statistics(statistics)


def list_pkl_files(directory="outputs"):
    """列出所有pkl文件"""
    if not os.path.exists(directory):
        print(f"❌ 目录 {directory} 不存在")
        return []
    
    pkl_pattern = os.path.join(directory, "*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"❌ 在 {directory} 目录中未找到pkl文件")
        return []
    
    # 按文件名排序
    pkl_files.sort()
    return pkl_files


def main():
    print("🔍 DeepConf PKL 文件读取器")
    print("=" * 50)
    
    # 查找pkl文件
    pkl_files = list_pkl_files()
    
    if not pkl_files:
        return
    
    print(f"\n📂 找到 {len(pkl_files)} 个pkl文件:")
    for i, pkl_file in enumerate(pkl_files, 1):
        filename = os.path.basename(pkl_file)
        file_size = os.path.getsize(pkl_file) / 1024  # KB
        print(f"  {i}. {filename} ({file_size:.1f} KB)")
    
    # 读取所有文件或让用户选择
    if len(pkl_files) == 1:
        # 只有一个文件，直接读取
        data = read_pkl_file(pkl_files[0])
        if data:
            print_pkl_summary(data, os.path.basename(pkl_files[0]))
    else:
        # 多个文件，让用户选择
        print(f"\n请选择要读取的文件 (1-{len(pkl_files)}, 或输入 'all' 读取所有文件):")
        choice = input("选择: ").strip().lower()
        
        if choice == 'all':
            # 读取所有文件
            for pkl_file in pkl_files:
                data = read_pkl_file(pkl_file)
                if data:
                    print_pkl_summary(data, os.path.basename(pkl_file))
                    print()  # 文件之间空行分隔
        else:
            try:
                file_index = int(choice) - 1
                if 0 <= file_index < len(pkl_files):
                    data = read_pkl_file(pkl_files[file_index])
                    if data:
                        print_pkl_summary(data, os.path.basename(pkl_files[file_index]))
                else:
                    print("❌ 无效的文件索引")
            except ValueError:
                print("❌ 请输入有效的数字或 'all'")


if __name__ == "__main__":
    main()
