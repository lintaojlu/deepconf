import openai
import json
import numpy as np
from collections import Counter

# ===========================
# Configuration
# ===========================
MODEL_PATH = "p2q"
MAX_TOKENS = 12800
PORT = 8000

# Inference parameters
WARMUP_TRACES = 16
TOTAL_BUDGET = 32
CONFIDENCE_PERCENTILE = 90
WINDOW_SIZE = 2048

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
# Trace Processing
# ===========================
def process_trace(choice, trace_id):
    """Process a single trace and extract confidence"""
    text = choice.message.content
    tokens = [t.token for t in choice.logprobs.content]

    # Calculate confidence
    confs = compute_confidence([t.top_logprobs for t in choice.logprobs.content])
    sliding_window = compute_least_grouped(confs, group_size=WINDOW_SIZE)

    trace_data = {
        "trace_id": trace_id,
        "text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "confs": confs,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
    }

    return trace_data

# ===========================
# Main Inference Function
# ===========================
def deepconf_inference(query):
    """Perform deepconf inference with warmup and final phases"""
    
    print("="*60)
    print("DEEPCONF INFERENCE")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Query: {query}")
    print(f"Warmup traces: {WARMUP_TRACES}")
    print(f"Total budget: {TOTAL_BUDGET}")
    print(f"Confidence percentile: {CONFIDENCE_PERCENTILE}")

    # Initialize client
    client = openai.OpenAI(
        api_key="None",
        base_url=f"http://localhost:{PORT}/v1",
        timeout=None
    )

    # System prompt for deepconf
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
4. 不允许丢掉任何关键词和限定词，不允许添加无意义的词和指代词，如"是什么"、"是谁"、"这"等。
5. 在拆解中，可以根据用户当前问题的核心意图类别和所属领域类别，参考上文的拆解方向进行拆解。
6. 在拆解中，首项必须为原始query的完整规范化总结，若首项与用户问题完全相同则改为"<same>"。例如："贵州2023-2025GDP"应拆解为"<same>;贵州2023年GDP数据;贵州2024年GDP数据;贵州2025年GDP数据"，"结合财报分析众安在线的业务亮点，以及解释近期股价的暴涨原因"应拆解为"结合财报分析众安在线的业务亮点并解释近期股价暴涨的原因;结合财报分析众安在线的业务亮点;众安在线财报中的业务亮点分析;解释众安在线近期股价暴涨的原因;众安在线近期股价暴涨的具体驱动因素"
7. 当用户问题中出现模糊时间时，需要参考默认时间修改为准确时间，如今天、明天等。
8. 拆解不允许修改英文专有名词。
9. 当用户问题中出现多个对比时，第一个拆解需要保持完整语意，其他后续拆解需要为各个的信息查询。例如："vivoiqooneo10Pro+和vivoiqooneo9sPro超详细比对"应拆解为"vivoiqooneo10Pro+和vivoiqooneo9sPro对比;vivoiqooneo10Pro+参数;vivoiqooneo9sPro参数;vivoiqooneo10Pro+性能;vivoiqooneo9sPro性能"
10. 拆解必须遵循用户当前问题出现的内容，不要进行过多扩展。
11. 每个拆解都是独立的拆解语句，当用户有多重条件时，每个拆解均需要保留所有前置条件，尤其是用户当前问题超长时，不要出现依赖与其他拆解query的拆解。
12. 拆解数量不要过多，尽量不超过8个。
13. 不要出现语意相似的拆解。
14. 只有个人能力、翻译、公式计算、打招呼、闲聊、代码解析、简单推理题、算命、起名、心理咨询情感沟通类等无需联网，其他问题如专有名词解释等均需要联网。
15. 默认位置为北京

~~~当前时间为：
2025年9月3日18:00:00
~~~

输出答案严格按照以下格式：
<net>是否联网</net><dismantle>拆解内容</dismantle>
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户:{query}"}
    ]

    # ===========================
    # WARMUP PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("WARMUP PHASE")
    print(f"{'-'*40}")

    warmup_request_params = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "logprobs": True,
        "top_logprobs": 20,
        "n": WARMUP_TRACES,
        "extra_body": {"top_k": 0},
    }
    
    responses = client.chat.completions.create(**warmup_request_params)

    # Process warmup traces
    warmup_traces = []
    min_confs = []

    for j in range(WARMUP_TRACES):
        trace_data = process_trace(responses.choices[j], j)
        warmup_traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])

    # Calculate confidence bar
    conf_bar = float(np.percentile(min_confs, CONFIDENCE_PERCENTILE))

    print(f"Confidence bar (P{CONFIDENCE_PERCENTILE}): {conf_bar:.4f}")
    print(f"Warmup traces generated: {len(warmup_traces)}")

    # Show some example results
    print(f"\nFirst 3 warmup traces:")
    for i, trace in enumerate(warmup_traces[:3]):
        print(f"  Trace {i}: conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']}")

    # ===========================
    # FINAL PHASE
    # ===========================
    print(f"\n{'-'*40}")
    print("FINAL PHASE (with early stopping)")
    print(f"{'-'*40}")

    real_gen = TOTAL_BUDGET - WARMUP_TRACES

    final_request_params = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "logprobs": True,
        "top_logprobs": 20,
        "n": real_gen,
        "extra_body": {
            "top_k": 0,
            "vllm_xargs": {
                'enable_conf': True,
                'window_size': WINDOW_SIZE,
                'threshold': conf_bar
            }
        }
    }
    
    responses = client.chat.completions.create(**final_request_params)
    
    # Process final traces
    final_traces = []
    for j in range(len(responses.choices)):
        trace_data = process_trace(responses.choices[j], WARMUP_TRACES + j)
        final_traces.append(trace_data)

    print(f"Final traces generated: {len(final_traces)} (requested: {real_gen})")

    # Show some example results
    print(f"\nFirst 3 final traces:")
    for i, trace in enumerate(final_traces[:3]):
        print(f"  Trace {i}: conf: {trace['min_conf']:.4f}, tokens: {trace['token_count']}")

    # ===========================
    # VOTING FOR FINAL ANSWER
    # ===========================
    print(f"\n{'-'*40}")
    print("VOTING FOR FINAL ANSWER")
    print(f"{'-'*40}")

    all_traces = warmup_traces + final_traces

    # Collect answers above threshold for voting
    voting_answers = []
    voting_weights = []

    # Add warmup traces above threshold (use confidence as weight)
    for trace in warmup_traces:
        minx = trace['min_conf']
        if minx >= conf_bar:
            answer = trace['text']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(minx)

    # Add final traces above threshold (use weight 1)
    for trace in final_traces:
        minx = trace['min_conf']
        if minx >= conf_bar:
            answer = trace['text']
            if answer is not None:
                voting_answers.append(answer)
                voting_weights.append(1.0)

    print(f"Traces used for voting: {len(voting_answers)}")

    # Perform weighted majority vote
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)

    print(f"Voted answer: {voted_answer}")

    # Show voting breakdown
    if voting_answers:
        answer_counts = Counter(voting_answers)
        print(f"\nVoting breakdown:")
        for answer, count in answer_counts.most_common():
            print(f"  {answer[:100]}...: {count} votes")

    # ===========================
    # FINAL SUMMARY
    # ===========================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Confidence bar: {conf_bar:.4f}")
    print(f"Total traces generated: {len(all_traces)}")
    print(f"Traces used for voting: {len(voting_answers)}")
    print(f"Final answer: {voted_answer}")

    return voted_answer

# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    # Define your query here
    query = "如何学习Python编程？"
    
    # Run deepconf inference
    result = deepconf_inference(query)
    
    print(f"\n🎯 Final Result: {result}")
