#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
包含各种实用的文件处理和数据转换函数
"""

import pandas as pd
import os
from typing import Optional


def csv_to_xlsx(csv_file_path: str, xlsx_file_path: Optional[str] = None, sheet_name: str = "Sheet1") -> str:
    """
    将CSV文件转换为XLSX文件
    
    Args:
        csv_file_path (str): CSV文件的路径
        xlsx_file_path (str, optional): 输出XLSX文件的路径。如果为None，则自动生成
        sheet_name (str): Excel工作表的名称，默认为"Sheet1"
    
    Returns:
        str: 生成的XLSX文件路径
    
    Raises:
        FileNotFoundError: 如果CSV文件不存在
        ValueError: 如果文件路径无效
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
    
    # 检查文件扩展名
    if not csv_file_path.lower().endswith('.csv'):
        raise ValueError(f"输入文件必须是CSV格式: {csv_file_path}")
    
    # 如果没有指定输出路径，自动生成
    if xlsx_file_path is None:
        # 获取CSV文件的目录和文件名（不含扩展名）
        csv_dir = os.path.dirname(csv_file_path)
        csv_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        xlsx_file_path = os.path.join(csv_dir, f"{csv_name}.xlsx")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(xlsx_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        # 写入XLSX文件
        print(f"正在转换为XLSX文件: {xlsx_file_path}")
        with pd.ExcelWriter(xlsx_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ 转换完成！")
        print(f"   CSV文件: {csv_file_path}")
        print(f"   XLSX文件: {xlsx_file_path}")
        print(f"   数据行数: {len(df)}")
        print(f"   数据列数: {len(df.columns)}")
        
        return xlsx_file_path
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        raise


def xlsx_to_csv(xlsx_file_path: str, csv_file_path: Optional[str] = None, sheet_name: Optional[str] = None) -> str:
    """
    将XLSX文件转换为CSV文件
    
    Args:
        xlsx_file_path (str): XLSX文件的路径
        csv_file_path (str, optional): 输出CSV文件的路径。如果为None，则自动生成
        sheet_name (str, optional): 要转换的工作表名称。如果为None，则转换第一个工作表
    
    Returns:
        str: 生成的CSV文件路径
    
    Raises:
        FileNotFoundError: 如果XLSX文件不存在
        ValueError: 如果文件路径无效或工作表不存在
    """
    # 检查XLSX文件是否存在
    if not os.path.exists(xlsx_file_path):
        raise FileNotFoundError(f"XLSX文件不存在: {xlsx_file_path}")
    
    # 检查文件扩展名
    if not xlsx_file_path.lower().endswith(('.xlsx', '.xls')):
        raise ValueError(f"输入文件必须是XLSX或XLS格式: {xlsx_file_path}")
    
    # 如果没有指定输出路径，自动生成
    if csv_file_path is None:
        # 获取XLSX文件的目录和文件名（不含扩展名）
        xlsx_dir = os.path.dirname(xlsx_file_path)
        xlsx_name = os.path.splitext(os.path.basename(xlsx_file_path))[0]
        csv_file_path = os.path.join(xlsx_dir, f"{xlsx_name}.csv")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(csv_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取XLSX文件
        print(f"正在读取XLSX文件: {xlsx_file_path}")
        
        # 如果没有指定工作表名称，读取第一个工作表
        if sheet_name is None:
            df = pd.read_excel(xlsx_file_path, engine='openpyxl')
            print(f"正在转换第一个工作表")
        else:
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, engine='openpyxl')
            print(f"正在转换工作表: {sheet_name}")
        
        # 写入CSV文件
        print(f"正在转换为CSV文件: {csv_file_path}")
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        
        print(f"✅ 转换完成！")
        print(f"   XLSX文件: {xlsx_file_path}")
        print(f"   CSV文件: {csv_file_path}")
        print(f"   数据行数: {len(df)}")
        print(f"   数据列数: {len(df.columns)}")
        
        return csv_file_path
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        raise


def main():
    """主函数，演示CSV到XLSX和XLSX到CSV转换功能"""
    print("=" * 60)
    print("文件格式转换工具演示")
    print("=" * 60)
    
    # 演示CSV到XLSX转换
    print("\n1. CSV到XLSX转换演示:")
    csv_file_path = "./data/0901版本planner.csv"
    xlsx_file_path = "./data/0901版本planner.csv_out_0~1000_test.xlsx"
    xlsx_to_csv(xlsx_file_path, csv_file_path)


if __name__ == "__main__":
    main()
