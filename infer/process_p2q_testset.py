import pandas as pd
import json
import os

def excel_to_jsonl(excel_path, output_path):
    """
    将Excel文件转换为JSONL格式

    Args:
        excel_path (str): Excel文件路径
        output_path (str): 输出JSONL文件路径
    """
    # 读取Excel文件
    print(f"正在读取Excel文件: {excel_path}")
    df = pd.read_excel(excel_path)

    print(f"原始数据行数: {len(df)}")

    # 显示策略gsb分布统计信息
    print(f"策略gsb分布（原始数据）: {df['策略gsb'].value_counts(dropna=False).to_dict()}")

    # 检查必要的列是否存在
    required_columns = ['query', '联网判断', 'dismantle', 'teg_subqs', '策略gsb', '策略打分']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")

    # 处理数据并生成JSONL格式
    jsonl_data = []

    for index, row in df.iterrows():
        # 获取question（query列）
        question = str(row['query']).strip() if pd.notna(row['query']) else ""

        # 获取联网判断列的值
        net_value = str(row['联网判断']).strip() if pd.notna(row['联网判断']) else ""

        # 获取策略gsb和策略打分
        gsb = str(row['策略gsb']).strip() if pd.notna(row['策略gsb']) else ""
        score = row['策略打分'] if pd.notna(row['策略打分']) else None

        # 处理answer内容
        if score is not None and score >= 2:
            if gsb in ['G', 'S']:
                dismantle_value = str(row['dismantle']).strip() if pd.notna(row['dismantle']) else ""
            elif gsb == 'B' or gsb == '':
                dismantle_value = str(row['teg_subqs']).strip() if pd.notna(row['teg_subqs']) else ""
            else:
                dismantle_value = ""
            answer = f"<net>{net_value}</net><dismantle>{dismantle_value}</dismantle>"
        else:
            # 分数小于2，truth直接设置为空
            answer = ""

        # 创建JSON对象
        json_obj = {
            "question": question,
            "answer": answer
        }

        jsonl_data.append(json_obj)

    # 写入JSONL文件
    print(f"正在写入JSONL文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for json_obj in jsonl_data:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

    print(f"转换完成！共处理了 {len(jsonl_data)} 条记录")
    print(f"输出文件: {output_path}")

    return len(jsonl_data)

def main():
    # 设置文件路径
    excel_path = "data/0901版本planner.csv_out_0~1000_test.xlsx"
    output_path = "p2q_testset.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(excel_path):
        print(f"错误：Excel文件不存在: {excel_path}")
        return
    
    try:
        # 执行转换
        record_count = excel_to_jsonl(excel_path, output_path)
        
        # 显示前几条记录作为预览
        print("\n前5条记录预览:")
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                data = json.loads(line)
                print(f"Record {i+1}:")
                print(f"  Question: {data['question'][:100]}{'...' if len(data['question']) > 100 else ''}")
                print(f"  Answer: {data['answer'][:100]}{'...' if len(data['answer']) > 100 else ''}")
                print()
                
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()
