import openai
import json
from tqdm import tqdm
import time
import os
from datetime import datetime
import numpy as np
from collections import Counter
import pickle
# from  dynasor.core.evaluator import math_equal
import re
import pandas as pd
from sentence_transformer_utils import SentenceTransformerSimilarity

# ===========================
# Configuration
# ===========================
MODEL_PATH = "p2q"
MAX_TOKENS = 12800
RID = 0
QID = 0
PORT = 8000
DATASET_FILE = "data/p2q_testset.jsonl"

# Online algorithm parameters
WARMUP_TRACES = 16
TOTAL_BUDGET = 32
CONFIDENCE_PERCENTILE = 90
WINDOW_SIZE = 2048

# Global Sentence Transformer instance
_sentence_transformer = None

def get_sentence_transformer():
    """获取全局Sentence Transformer实例"""
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformerSimilarity()
        _sentence_transformer.download_model()
    return _sentence_transformer

# ===========================
# Edit Distance and Scoring Functions
# ===========================
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def extract_net_dismantle(text):
    """Extract net and dismantle parts from text"""
    if not text:
        return None, None
    
    # Pattern to match <net>content</net><dismantle>content</dismantle>
    pattern = r'<net>(.*?)</net><dismantle>(.*?)</dismantle>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        net_part = match.group(1).strip()
        dismantle_part = match.group(2).strip()
        return net_part, dismantle_part
    
    return None, None

def calculate_score(voted_answer, ground_truth):
    """
    Calculate score based on net and dismantle parts using Sentence Transformer
    net_score = 1 if net parts are same, else 0
    dismantle_score = best average score from all possible mappings between dismantle parts
    final_score = net_score * dismantle_score
    """
    if not voted_answer or not ground_truth:
        return 0.0, 0.0, 0.0
    
    # Extract net and dismantle parts
    voted_net, voted_dismantle = extract_net_dismantle(voted_answer)
    truth_net, truth_dismantle = extract_net_dismantle(ground_truth)
    
    # Calculate net score (binary: 1 if same, 0 if different)
    net_score = 1.0 if (voted_net and truth_net and voted_net == truth_net) else 0.0
    
    # Calculate dismantle score using optimal mapping
    dismantle_score = 0.0
    if voted_dismantle and truth_dismantle:
        dismantle_score = calculate_dismantle_mapping_score(voted_dismantle, truth_dismantle)
    elif not voted_dismantle and not truth_dismantle:
        dismantle_score = 1.0  # Both empty, perfect match
    
    # Calculate final score
    final_score = net_score * dismantle_score
    
    return net_score, dismantle_score, final_score


def calculate_dismantle_mapping_score(voted_dismantle, truth_dismantle):
    """
    Calculate dismantle score by finding the best mapping between dismantle parts
    Split by semicolon and find the optimal one-to-one mapping using batch similarity computation
    """
    # Split by semicolon and strip whitespace
    voted_parts = [part.strip() for part in voted_dismantle.split(';') if part.strip()]
    truth_parts = [part.strip() for part in truth_dismantle.split(';') if part.strip()]
    
    # If either is empty, return 0
    if not voted_parts or not truth_parts:
        return 0.0
    
    # If lengths are different, we need to handle all possible mappings
    # For simplicity, we'll use the shorter length and find best mapping
    min_len = min(len(voted_parts), len(truth_parts))
    
    if min_len == 0:
        return 0.0
    
    # Get Sentence Transformer instance
    sentence_transformer = get_sentence_transformer()
    
    # Generate all possible permutations of truth_parts
    from itertools import permutations
    
    best_avg_score = 0.0
    
    # Collect all text pairs for batch processing
    all_text_pairs = []
    permutation_indices = []
    
    for perm_idx, perm in enumerate(permutations(truth_parts, min_len)):
        perm_start_idx = len(all_text_pairs)
        
        # Collect text pairs for this permutation
        for i in range(min_len):
            voted_part = voted_parts[i]
            truth_part = perm[i]
            
            if voted_part and truth_part:
                all_text_pairs.append((voted_part, truth_part))
        
        # Record which pairs belong to this permutation
        perm_end_idx = len(all_text_pairs)
        permutation_indices.append((perm_start_idx, perm_end_idx))
    
    # Batch compute all similarities at once
    if all_text_pairs:
        all_similarities = sentence_transformer.compute_pairs_similarity(all_text_pairs)
        
        # Process results for each permutation
        for perm_idx, (start_idx, end_idx) in enumerate(permutation_indices):
            perm_similarities = all_similarities[start_idx:end_idx]
            
            if perm_similarities:
                avg_score = sum(perm_similarities) / len(perm_similarities)
                best_avg_score = max(best_avg_score, avg_score)
    
    return best_avg_score

# ===========================
# Answer Extraction Functions
# ===========================
def extract_answer(text):
    """Extract answer from text - supports both boxed format and net/dismantle format"""
    if not text:
        return None
    
    # First try to extract net/dismantle format (our target format)
    net_dismantle_pattern = r'<net>.*?</net><dismantle>.*?</dismantle>'
    match = re.search(net_dismantle_pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

# ===========================
# Data Loading
# ===========================
def load_aime25_jsonl(file_path=DATASET_FILE):
    """Load data from aime25.jsonl file"""
    print(f"Loading data from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    print(f"Skipping empty line {line_num}")
                    continue
                try:
                    parsed_line = json.loads(line)
                    data.append(parsed_line)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {line_num}: {e}")
                    print(f"Line content: {repr(line[:100])}")
                    print(f"Line length: {len(line)}")
                    raise
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise
    
    print(f"Successfully loaded {len(data)} items")
    return data

# ===========================
# Confidence Calculation
# ===========================
def compute_confidence(logprobs):
    """Compute confidence score from logprobs."""
    confs = []
    for lp in logprobs:
        confs.append(round(-sum([l.logprob for l in lp]) / len(lp), 3))
    return confs

def compute_least_grouped(confs, group_size=WINDOW_SIZE):
    """Compute sliding window mean confidence with specified group size."""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]

    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))

    return sliding_means

# ===========================
# Voting Functions
# ===========================
def weighted_majority_vote(answers, weights):
    """Perform weighted majority voting"""
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return voted_answer

# ===========================
# Trace Processing and Accuracy Calculation
# ===========================
def process_trace(choice, trace_id, ground_truth):
    """Process a single trace and calculate accuracy"""
    # Extract basic info
    text = choice.message.content
    tokens = [t.token for t in choice.logprobs.content]

    # Calculate confidence
    confs = compute_confidence([t.top_logprobs for t in choice.logprobs.content])
    sliding_window = compute_least_grouped(confs, group_size=WINDOW_SIZE)

    # Extract answer
    extracted_answer = extract_answer(text)

    # Calculate correctness using our custom scoring function
    final_score = 0.0
    if extracted_answer is not None and ground_truth is not None:
        _, _, final_score = calculate_score(extracted_answer, ground_truth)

    trace_data = {
        "trace_id": trace_id,
        "stop_reason": choice.stop_reason,
        "finish_reason": choice.finish_reason,
        "text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "confs": confs,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
        "final_score": final_score,
    }

    return trace_data

def calculate_statistics(traces, phase_name=""):
    """Calculate statistics for a list of traces"""
    if not traces:
        return {}

    total_traces = len(traces)
    total_tokens = sum(t['token_count'] for t in traces)

    # Confidence statistics
    min_confs = [t['min_conf'] for t in traces]
    final_scores = [t['final_score'] for t in traces]
    stats = {
        f"{phase_name}_traces": total_traces,
        f"{phase_name}_final_score_mean": np.mean(final_scores) if final_scores else 0,
        f"{phase_name}_final_score_std": np.std(final_scores) if final_scores else 0,
        f"{phase_name}_total_tokens": total_tokens,
        f"{phase_name}_avg_tokens_per_trace": total_tokens / total_traces if total_traces > 0 else 0,
        f"{phase_name}_min_conf_mean": np.mean(min_confs) if min_confs else 0,
        f"{phase_name}_min_conf_std": np.std(min_confs) if min_confs else 0,
    }

    return stats

# ===========================
# Process Single Question Function
# ===========================
def process_single_question(data, qid, run_id=0):
    """Process a single question and return results"""
    print("="*60)
    print("ONLINE ALGORITHM WITH ACCURACY CALCULATION")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Question ID: {qid}")
    print(f"Run ID: {run_id}")
    print(f"Warmup traces: {WARMUP_TRACES}")
    print(f"Total budget: {TOTAL_BUDGET}")
    print(f"Confidence percentile: {CONFIDENCE_PERCENTILE}")

    # Initialize client
    client = openai.OpenAI(
        api_key="None",
        base_url=f"http://localhost:{PORT}/v1",
        timeout=None
    )

    # Get question and ground truth
    prompt = data[qid]['question']
    ground_truth = str(data[qid].get('answer', '')).strip()

    system_prompt = """你是元宝。观察助手和用户的历史对话,输出当前问题的改写，包括联网和拆解。

## 用户问题类别：
用户的问题可能属于以下两个维度的类别之一。在拆解时，你需要先判断用户问题的类别，并参考对应类别的拆解方向进行拆解。

### 1. 核心意图类型：
- 事实查询（What/Who）→ 定义与概念拆解
- 原因查询（Why）→ 原因与解决方案拆解
- 方法查询（How）→ 方法与解决方案拆解
- 比较查询（vs/difference）→ 比较与差异拆解
- 建议查询（should/recommendation）→ 策略与方案拆解
- 其他类别

### 2. 所属领域类型及其拆解方向：
- 医疗健康领域 → 症状+原因+处理方法
- 技术产品领域 → 简介+功能+使用方法
- 金融投资领域 → 策略+风险+操作建议
- 法律领域 → 规定+后果+案例
- 景点旅游类 → 票务时间 + 攻略
- 美食（店铺）类 → 特色
- 教育教学  → 知识点（定义 / 公式 / 考点）+方法（技巧 / 计划）+考试信息（时间 / 题型 / 分数线）+教材 / 资料推荐+升学政策+培训机构选择等
- 职场求职 → 岗位职责、招聘要求（学历 / 技能）+面试技巧（常见问题 / 应答策略）+简历优化方法+薪资范围+行业前景+离职 / 跳槽注意事项
- 情感心理 → 情绪表现（焦虑 / 抑郁）+成因分析（家庭 / 工作）+调节方法（沟通 / 运动）+关系处理（情侣 / 亲子）+专业咨询建议
- 其他领域(可适当扩展)

## 注意：
1. 你需要先识别出用户当前问题的核心意图类别和所属领域类别，以便用户后续的拆解。
2. 请务必保证当前轮次的用户问题足够完整，拆解的指代词都完整替换了，且拆解必须明确，不允许出现模糊词。当用户基于对话日志进行提问时，需要将对话日志中的答案补充到拆解。
3、拆解必须为针对用户查询拆解的查询语句，适合用于外部知识检索以补充LLM生成答案所需信息。
4. 不允许丢掉任何关键词和限定词，不允许添加无意义的词和指代词，如“是什么”、“是谁”、“这”等。
5. 在拆解中，可以根据用户当前问题的核心意图类别和所属领域类别，参考上文的拆解方向进行拆解。
6. 在拆解中，首项必须为原始query的完整规范化总结，若首项与用户问题完全相同则改为“<same>”。例如：“贵州2023-2025GDP”应拆解为“<same>;贵州2023年GDP数据;贵州2024年GDP数据;贵州2025年GDP数据”，“结合财报分析众安在线的业务亮点，以及解释近期股价的暴涨原因”应拆解为“结合财报分析众安在线的业务亮点并解释近期股价的暴涨原因;结合财报分析众安在线的业务亮点;众安在线财报中的业务亮点分析;解释众安在线近期股价暴涨的原因;众安在线近期股价暴涨的具体驱动因素”
7. 当用户问题中出现模糊时间时，需要参考默认时间修改为准确时间，如今天、明天等。
8. 拆解不允许修改英文专有名词。
9. 当用户问题中出现多个对比时，第一个拆解需要保持完整语意，其他后续拆解需要为各个的信息查询。例如：“vivoiqooneo10Pro+和vivoiqooneo9sPro超详细比对”应拆解为“vivoiqooneo10Pro+和vivoiqooneo9sPro对比;vivoiqooneo10Pro+参数;vivoiqooneo9sPro参数;vivoiqooneo10Pro+性能;vivoiqooneo9sPro性能”
10. 拆解必须遵循用户当前问题出现的内容，不要进行过多扩展。
11. 每个拆解都是独立的拆解语句，当用户有多重条件时，每个拆解均需要保留所有前置条件，尤其是用户当前问题超长时，不要出现依赖与其他拆解query的拆解。
12. 拆解数量不要过多，尽量不超过8个。
13. 不要出现语意相似的拆解。
14. 只有个人能力、翻译、公式计算、打招呼、闲聊、代码解析、简单推理题、算命、起名、心理咨询情感沟通类等无需联网，其他问题如专有名词解释等均需要联网。
15. 默认位置为北京

## 时间转换规则
需要将用户输入中的时间相关表达（如“本周”、“今年”、“近几年”等）准确映射为具体的时间范围或时间点。请严格按照以下规则进行解析和映射：

### 1. 时效pattern解析规则

| 时间表达         | 映射规则（范围/点）                 | 示例（以2025年6月1日为例）                   |
|------------------|-------------------------------------|-----------------------------------------------|
| 本周/这周/最近一周 | 本周周一至周日以及本周每一天        | 2025年6月1日-2025年6月7日，2025年6月1日，2025年6月2日，2025年6月3日，2025年6月4日，2025年6月5日，2025年6月6日，2025年6月7日|
| 本月             | 当前自然月                          | 2025年6月                                     |
| 今日             | 当前日期                            | 2025年6月1日                                  |
| 今年             | 当前年份                            | 2025年                                        |
| 去年             | 上一个年份                           | 2024年                                        |
| 近几天/最近几天    | 当前日期往前推7天以及该范围的每一天   | 2025年5月26日-2025年6月1日，2025年5月26日，2025年5月27日，2025年5月28日，2025年5月29日，2025年5月30日，2025年5月31日，2025年6月1日|
| 近几月           | 当前月及往前推若干月                | 2025年6月，2025年5月，2024年4月               |
| 近几年           | 当前年及往前推若干年                | 2025年，2024年，2023年                        |
| 近两年           | 当前年及上一个年份                  | 2025年，2024年                                |
| 历年             | 当前年及所有往年                    | 2025年，2024年，2023年                        |
| 历年今日         | 当前日期及所有往年同日              | 2025年6月1日，2024年6月1日，2023年6月1日       |
| 本季度/本Q季度   | 当前季度所有月份                    | 2025年4月，2025年5月，2025年6月，2025年Q2，第二季度  |
| 上季度/上一季度  | 上一季度所有月份                    | 2025年1月，2025年2月，2025年3月，2025年Q1，第一季度   |
| 上个月/上月       | 上一个月份                           | 2025年5月                                     |
| 上周             | 上周周一至周日以及上周每天          | 2025年5月25日-2025年5月31日，2025年5月25日，2025年5月26日，2025年5月27日，2025年5月28日，2025年5月29日，2025年5月30日，2025年5月31日|
| 近期             | 当前月                              | 2025年6月                                     |
| 最新             | 当前月                              | 2025年6月                                     |
| 最近             | 当前月                              | 2025年6月                                     |
| 现在             | 当前日期                            | 2025年6月1日                                  |
| 明天/明日        | 当前日期+1天                        | 2025年6月2日                                  |
| 未来一周         | 当前日期至+7天                  | 2025年6月1日-2025年6月7日，2025年6月1日，2025年6月2日，2025年6月3日，2025年6月4日，2025年6月5日，2025年6月6日，2025年6月7日|
| 未来一个月       | 当前日期+1天至+1月以及下一个月份        | 2025年6月2日-2025年7月2日，2025年6月            |
| 下周             | 下一个自然周以及下周的每一天          | 2025年6月8日-2025年6月14日，2025年6月8日，2025年6月9日，2025年6月10日，2025年6月11日，2025年6月12日，2025年6月13日，2025年6月14日 |
| 最近**个交易日   | 当前日期往前推指定交易日数及该范围的每一天   | 2025年5月30日-2025年6月1日（近3个交易日），2025年5月30日，2025年5月31日，2025年6月1日|
| **股票**/走势/价格 | 当天或当月                         | 2025年6月1日（当天）、2025年6月（当月）        |
| **应季**         | 当前月                              | 2025年6月                                     |
| **目前**         | 当前年                              | 2025年                                        |
| 政策/数据/房价   | 当前月或年，视具体领域而定           | 2025年6月或2025年                             |
| 中考/高考        | 当前年                              | 2025年                                        |

### 2. 排行榜时效补全规则

| 频率      | 补全指令词 | 示例                          | 特征说明                          |
|-----------|------------|-------------------------------|-----------------------------------|
| 年度更新  | 2025       | QS世界大学排名、公司年报等    | 权威性强，覆盖宏观评估            |
| 月/周更新 | 2025年6月  | App Store下载榜、月度榜单等   | 反映中短期趋势                    |
| 天级/实时 | 2025年6月1日 | 热搜榜、即时榜单等           | 捕捉即时动态                      |
| 季度/半年 | 2025年     | 小米季度报、企业财报季榜单等  | 财报、部分专业领域榜单            |

### 3. 输出要求

- 请将用户输入中的时间表达，统一映射为具体的时间点或时间段（以“年”、“月”、“日”为单位）。
- 如遇不明确表达，优先结合语境及上述规则合理推断。
- 若涉及排行榜、榜单、政策、财报等，结合其更新频率补全时间粒度。
- 除了要解析出具体的时间范围，还要拆分到每一天或每个月或每一年。
   例如：‘2025年3月10号至2025年3月13号的足球比赛’需要拆分为‘<same>;2025年3月10号的足球比赛;2025年3月11号的足球比赛;2025年3月12号的足球比赛;2025年3月13号的足球比赛‘
   例如：‘2025年3月至5月的金价’需要拆分为‘<same>;2025年3月金价;2025年4月金价;2025年5月金价’
   例如：‘2025年-2026年的江苏GDP’需要拆分为‘<same>;2025年的江苏GDP;2026年的江苏GDP’

~~~当前时间为：
2025年9月3日18:00:00
~~~

输出答案严格按照以下格式：
<net>是否联网</net><dismantle>拆解内容</dismantle>
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户:{prompt}"}
    ]

    print(f"\nQuestion: {prompt}")
    print(f"Ground Truth: {ground_truth}")

    # ===========================
    # WARMUP PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("WARMUP PHASE")
    print(f"{'-'*40}")

    t0 = time.time()
    responses = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        logprobs=True,
        top_logprobs=20,
        n=WARMUP_TRACES,
        extra_body={"top_k": 0},
    )
    t1 = time.time()

    # Process warmup traces
    warmup_traces = []
    min_confs = []

    for j in range(WARMUP_TRACES):
        trace_data = process_trace(responses.choices[j], j, ground_truth)
        warmup_traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])

    # Calculate confidence bar
    conf_bar = float(np.percentile(min_confs, CONFIDENCE_PERCENTILE))

    # Calculate warmup statistics
    warmup_stats = calculate_statistics(warmup_traces, "warmup")

    print(f"Warmup time: {t1 - t0:.2f}s")
    print(f"Confidence bar (P{CONFIDENCE_PERCENTILE}): {conf_bar:.4f}")
    print(f"Warmup final score mean: {warmup_stats['warmup_final_score_mean']:.4f}")
    print(f"Warmup final score std: {warmup_stats['warmup_final_score_std']:.4f}")
    print(f"Warmup total tokens: {warmup_stats['warmup_total_tokens']}")
    print(f"Warmup avg tokens per trace: {warmup_stats['warmup_avg_tokens_per_trace']:.1f}")

    # Show some example results
    print(f"\nFirst 3 warmup traces:")
    for i, trace in enumerate(warmup_traces[:3]):
        print(f"  Trace {i}: {trace['extracted_answer']} (score: {trace['final_score']:.4f}, conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']})")

    # ===========================
    # FINAL PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("FINAL PHASE (with early stopping)")
    print(f"{'-'*40}")

    real_gen = TOTAL_BUDGET - WARMUP_TRACES

    t3 = time.time()
    responses = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        logprobs=True,
        top_logprobs=20,
        n=real_gen,
        extra_body={
            "top_k": 0,
            "vllm_xargs": {
                'enable_conf': True,
                'window_size': WINDOW_SIZE,
                'threshold': conf_bar
            }
        }
    )
    t4 = time.time()

    # Process final traces
    final_traces = []
    for j in range(len(responses.choices)):
        trace_data = process_trace(responses.choices[j], WARMUP_TRACES + j, ground_truth)
        final_traces.append(trace_data)

    # Calculate final statistics
    final_stats = calculate_statistics(final_traces, "final")

    print(f"Final time: {t4 - t3:.2f}s")
    print(f"Final traces generated: {len(final_traces)} (requested: {real_gen})")
    print(f"Final final score mean: {final_stats['final_final_score_mean']:.4f}")
    print(f"Final final score std: {final_stats['final_final_score_std']:.4f}")
    print(f"Final total tokens: {final_stats['final_total_tokens']}")
    print(f"Final avg tokens per trace: {final_stats['final_avg_tokens_per_trace']:.1f}")

    # Show some example results
    print(f"\nFirst 3 final traces:")
    for i, trace in enumerate(final_traces[:3]):
        print(f"  Trace {i}: {trace['extracted_answer']} (score: {trace['final_score']:.4f}, conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']})")

    # ===========================
    # OVERALL STATISTICS
    # ===========================
    print(f"\n{'-'*40}")
    print("OVERALL STATISTICS")
    print(f"{'-'*40}")

    all_traces = warmup_traces + final_traces
    overall_stats = calculate_statistics(all_traces, "overall")

    total_time = t4 - t0
    warmup_time = t1 - t0
    final_time = t4 - t3

    print(f"Total time: {total_time:.2f}s (warmup: {warmup_time:.2f}s, final: {final_time:.2f}s)")
    print(f"Overall traces: {len(all_traces)}")
    print(f"Overall final score mean: {overall_stats['overall_final_score_mean']:.4f}")
    print(f"Overall final score std: {overall_stats['overall_final_score_std']:.4f}")
    print(f"Overall total tokens: {overall_stats['overall_total_tokens']}")
    print(f"Overall avg tokens per trace: {overall_stats['overall_avg_tokens_per_trace']:.1f}")

    # Efficiency metrics
    tokens_per_second = overall_stats['overall_total_tokens'] / total_time
    traces_per_second = len(all_traces) / total_time

    print(f"Tokens per second: {tokens_per_second:.1f}")
    print(f"Traces per second: {traces_per_second:.2f}")

    # Early stopping analysis
    if final_traces:
        above_threshold = sum(1 for t in final_traces if t['min_conf'] >= conf_bar)
        print(f"Final traces above threshold: {above_threshold}/{len(final_traces)} ({above_threshold/len(final_traces):.2%})")

    # ===========================
    # SAVE RESULTS
    # ===========================
    results = {
        "question_id": qid,
        "run_id": run_id,
        "question": prompt,
        "ground_truth": ground_truth,
        "conf_bar": conf_bar,
        "warmup_traces": warmup_traces,
        "final_traces": final_traces,
        "statistics": {
            **warmup_stats,
            **final_stats,
            **overall_stats,
            "total_time": total_time,
            "warmup_time": warmup_time,
            "final_time": final_time,
            "tokens_per_second": tokens_per_second,
            "traces_per_second": traces_per_second,
        },
        "config": {
            "model_path": MODEL_PATH,
            "warmup_traces": WARMUP_TRACES,
            "total_budget": TOTAL_BUDGET,
            "confidence_percentile": CONFIDENCE_PERCENTILE,
            "window_size": WINDOW_SIZE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    pickle.dump(results, open(f"outputs/q{qid}_r{run_id}_{timestamp}.pkl", 'wb'))
    print(f"Results saved to outputs/q{qid}_r{run_id}_{timestamp}.pkl")
    output_filename = f"outputs/q{qid}_r{run_id}_{timestamp}.pkl"

    print(f"\n✅ Results saved to outputs/q{qid}_r{run_id}_{timestamp}.pkl")

    # ===========================
    # COMPARISON WITH BASELINE
    # ===========================
    print(f"\n{'-'*40}")
    print("COMPARISON ANALYSIS")
    print(f"{'-'*40}")

    # Compare warmup vs final phase
    if warmup_stats['warmup_traces'] > 0 and final_stats['final_traces'] > 0:
        score_improvement = final_stats['final_final_score_mean'] - warmup_stats['warmup_final_score_mean']
        print(f"Score change (final - warmup): {score_improvement:+.4f}")

        token_efficiency = final_stats['final_avg_tokens_per_trace'] / warmup_stats['warmup_avg_tokens_per_trace']
        print(f"Token efficiency (final/warmup): {token_efficiency:.2f}x")

    # Early stopping effectiveness
    total_requested = WARMUP_TRACES + (TOTAL_BUDGET - WARMUP_TRACES)
    total_generated = len(all_traces)
    saving_ratio = (total_requested - total_generated) / total_requested

    print(f"Traces requested: {total_requested}")
    print(f"Traces generated: {total_generated}")
    print(f"Early stopping saving: {saving_ratio:.2%}")

    # ===========================
    # VOTING FOR FINAL ANSWER
    # ===========================
    print(f"\n{'-'*40}")
    print("VOTING FOR FINAL ANSWER")
    print(f"{'-'*40}")

    # Collect answers above threshold for voting
    voting_answers = []
    voting_weights = []
    total_voting_tokens = 0

    # Add warmup traces above threshold (use confidence as weight)
    for trace in warmup_traces:
        minx = trace['min_conf']
        total_voting_tokens += trace['token_count']
        if minx >= conf_bar:
            answer = trace['extracted_answer']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(minx)

    # Add final traces above threshold (use weight 1)
    for trace in final_traces:
        total_voting_tokens += trace['token_count']
        # Skip traces that were stopped by gconf (early stopping)
        if trace['stop_reason'] is not None and 'gconf' in trace['stop_reason']:
            continue

        minx = trace['min_conf']
        # Note: final traces might not need threshold check since they were already filtered
        # But keeping it consistent with the reference code
        if minx >= conf_bar:
            answer = trace['extracted_answer']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(1)

    print(f"Traces used for voting: {len(voting_answers)}")
    print(f"Total tokens in voting traces: {total_voting_tokens}")

    # Perform weighted majority vote
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)

    # Calculate score based on net and dismantle parts
    net_score, dismantle_score, final_score = calculate_score(voted_answer, ground_truth)
    
    # For backward compatibility, we can still define is_voted_correct
    # but now it's based on the final_score (you can adjust threshold as needed)
    is_voted_correct = final_score > 0.8  # Threshold for considering as "correct"
    
    print(f"Voted answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Net score: {net_score:.4f}")
    print(f"Dismantle score: {dismantle_score:.4f}")
    print(f"Final score: {final_score:.4f}")
    print(f"Voted answer correct (threshold 0.8): {is_voted_correct}")

    # Show voting breakdown
    if voting_answers:
        answer_counts = Counter(voting_answers)
        print(f"\nVoting breakdown:")
        for answer, count in answer_counts.most_common():
            print(f"  {answer}: {count} votes")

    # ===========================
    # UPDATE RESULTS WITH VOTING
    # ===========================
    voting_results = {
        "voting_answers": voting_answers,
        "voting_weights": voting_weights,
        "voted_answer": voted_answer,
        "is_voted_correct": is_voted_correct,
        "net_score": net_score,
        "dismantle_score": dismantle_score,
        "final_score": final_score,
        "voting_traces_count": len(voting_answers),
        "voting_total_tokens": total_voting_tokens,
    }

    results["voting"] = voting_results
    results["statistics"]["voting_traces_count"] = len(voting_answers)
    results["statistics"]["voting_total_tokens"] = total_voting_tokens
    results["statistics"]["is_voted_correct"] = is_voted_correct
    results["statistics"]["net_score"] = net_score
    results["statistics"]["dismantle_score"] = dismantle_score
    results["statistics"]["final_score"] = final_score

    # Update the saved file with voting results
    # with open(output_filename, 'w', encoding='utf-8') as f:
    #     pickle.dump(results, f)

    # print(f"\n✅ Results updated with voting information: {output_filename}")

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Question ID: {qid}")
    print(f"Confidence bar: {conf_bar:.4f}")
    print(f"Total traces generated: {len(all_traces)}")
    print(f"Traces used for voting: {len(voting_answers)}")
    print(f"Total tokens: {overall_stats['overall_total_tokens']}")
    print(f"Voting answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Net score: {net_score:.4f}")
    print(f"Dismantle score: {dismantle_score:.4f}")
    print(f"Final score: {final_score:.4f}")
    print(f"Final result: {'✅ CORRECT' if is_voted_correct else '❌ INCORRECT'} (threshold: 0.8)")

    # Extract net and dismantle parts from voted answer for CSV output
    voted_net, voted_dismantle = extract_net_dismantle(voted_answer) if voted_answer else (None, None)
    
    return {
        "query": prompt,
        "truth": ground_truth,
        "answer": voted_answer,
        "net": voted_net,
        "dismantle": voted_dismantle,
        "net_score": net_score,
        "dismantle_score": dismantle_score,
        "final_score": final_score,
        "is_correct": is_voted_correct,
        "question_id": qid,
        "run_id": run_id,
        "total_traces": len(all_traces),
        "voting_traces": len(voting_answers),
        "total_tokens": overall_stats['overall_total_tokens']
    }

# ===========================
# Main Function
# ===========================
def main():
    """Main function to process all questions in the dataset"""
    print("="*60)
    print("BATCH PROCESSING ALL QUESTIONS")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"Warmup traces: {WARMUP_TRACES}")
    print(f"Total budget: {TOTAL_BUDGET}")
    print(f"Confidence percentile: {CONFIDENCE_PERCENTILE}")

    # Load the data
    data = load_aime25_jsonl()
    print(f"Loaded {len(data)} items from {DATASET_FILE}")

    # Initialize results list
    all_results = []
    
    # Process each question
    for qid in tqdm(range(len(data)), desc="Processing questions"):
        try:
            print(f"\n{'='*80}")
            print(f"Processing Question {qid + 1}/{len(data)}")
            print(f"{'='*80}")
            
            # Process single question
            result = process_single_question(data, qid, RID)
            all_results.append(result)
            
            print(f"✅ Completed question {qid + 1}/{len(data)}")
            
        except Exception as e:
            print(f"❌ Error processing question {qid}: {e}")
            # Add error result
            all_results.append({
                "query": data[qid].get('question', ''),
                "truth": data[qid].get('answer', ''),
                "answer": None,
                "net": None,
                "dismantle": None,
                "net_score": 0.0,
                "dismantle_score": 0.0,
                "final_score": 0.0,
                "is_correct": False,
                "question_id": qid,
                "run_id": RID,
                "total_traces": 0,
                "voting_traces": 0,
                "total_tokens": 0,
                "error": str(e)
            })
            continue

    # ===========================
    # Save Results to CSV
    # ===========================
    print(f"\n{'='*60}")
    print("SAVING RESULTS TO CSV")
    print(f"{'='*60}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Select columns for CSV output
    csv_columns = ['query', 'truth', 'answer', 'net', 'dismantle']
    csv_df = df[csv_columns].copy()
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"outputs/p2q_results_{timestamp}.csv"
    csv_df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"✅ CSV results saved to: {csv_filename}")
    print(f"📊 Processed {len(all_results)} questions")
    
    # Save detailed results
    detailed_filename = f"outputs/p2q_detailed_results_{timestamp}.csv"
    df.to_csv(detailed_filename, index=False, encoding='utf-8')
    print(f"✅ Detailed results saved to: {detailed_filename}")
    
    # ===========================
    # Summary Statistics
    # ===========================
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    successful_results = [r for r in all_results if 'error' not in r]
    error_count = len(all_results) - len(successful_results)
    
    if successful_results:
        avg_final_score = np.mean([r['final_score'] for r in successful_results])
        avg_net_score = np.mean([r['net_score'] for r in successful_results])
        avg_dismantle_score = np.mean([r['dismantle_score'] for r in successful_results])
        correct_count = sum([r['is_correct'] for r in successful_results])
        total_tokens = sum([r['total_tokens'] for r in successful_results])
        
        print(f"Total questions: {len(all_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Errors: {error_count}")
        print(f"Accuracy: {correct_count}/{len(successful_results)} ({correct_count/len(successful_results):.2%})")
        print(f"Average final score: {avg_final_score:.4f}")
        print(f"Average net score: {avg_net_score:.4f}")
        print(f"Average dismantle score: {avg_dismantle_score:.4f}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Average tokens per question: {total_tokens/len(successful_results):.0f}")
    
    return all_results

if __name__ == "__main__":
    results = main()