import openai
import json
import numpy as np
import pandas as pd
import re
from collections import Counter

# ===========================
# Configuration
# ===========================
MODEL_PATH = "p2q"
MAX_TOKENS = 12800
PORT = 8000

# Inference parameters
WARMUP_TRACES = 16
TOTAL_BUDGET = 64
CONFIDENCE_PERCENTILE = 10
WINDOW_SIZE = 8
TEMPERATURE = 0.9


# PROMPT配置 - 从文件读取
def load_prompt(prompt_file="data/prompt.txt"):
    """从文件中加载prompt配置"""
    import os

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, prompt_file)

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        prompt_config = {
            "system": system_prompt,
            "user": """助手:{ans}
用户:{query}""",
        }

        print(f"Loaded prompt from: {prompt_path}")
        return prompt_config

    except FileNotFoundError:
        print(f"Warning: Prompt file not found at {prompt_path}")
        print("Using default prompt...")

        # 使用默认prompt作为备选
        default_prompt = """你是元宝。观察助手和用户的历史对话,输出当前问题的改写，包括联网和拆解。

~~~当前时间为：
{time}
~~~

输出答案严格按照以下格式：
<net>是否联网</net><dismantle>拆解内容</dismantle>"""

        return {
            "system": default_prompt,
            "user": """助手:{ans}
用户:{query}""",
        }

    except Exception as e:
        print(f"Error loading prompt file: {e}")
        print("Using default prompt...")

        # 使用默认prompt作为备选
        default_prompt = """你是元宝。观察助手和用户的历史对话,输出当前问题的改写，包括联网和拆解。

~~~当前时间为：
{time}
~~~

输出答案严格按照以下格式：
<net>是否联网</net><dismantle>拆解内容</dismantle>"""

        return {
            "system": default_prompt,
            "user": """助手:{ans}
用户:{query}""",
        }


# 加载PROMPT配置
PROMPT = load_prompt()


# ===========================
# DeepConf推理类
# ===========================
class DeepConfInference:
    """封装DeepConf推理功能的类"""

    def __init__(
        self,
        model_path=MODEL_PATH,
        port=PORT,
        max_tokens=MAX_TOKENS,
        warmup_traces=WARMUP_TRACES,
        total_budget=TOTAL_BUDGET,
        confidence_percentile=CONFIDENCE_PERCENTILE,
        window_size=WINDOW_SIZE,
        temperature=TEMPERATURE,
    ):
        self.model_path = model_path
        self.port = port
        self.max_tokens = max_tokens
        self.warmup_traces = warmup_traces
        self.total_budget = total_budget
        self.confidence_percentile = confidence_percentile
        self.window_size = window_size
        self.temperature = temperature
        # 初始化客户端
        self.client = openai.OpenAI(
            api_key="None", base_url=f"http://localhost:{port}/v1", timeout=None
        )

    def compute_confidence(self, logprobs):
        """计算置信度分数"""
        confs = []
        for lp in logprobs:
            confs.append(round(-sum([l.logprob for l in lp]) / len(lp), 3))
        return confs

    def compute_least_grouped(self, confs, group_size=None):
        """计算滑动窗口平均置信度"""
        if group_size is None:
            group_size = self.window_size

        if len(confs) < group_size:
            return [sum(confs) / len(confs)] if confs else [0]

        sliding_means = []
        for i in range(len(confs) - group_size + 1):
            window = confs[i : i + group_size]
            sliding_means.append(round(sum(window) / len(window), 3))

        return sliding_means

    def weighted_majority_vote(self, answers, weights):
        """加权多数投票"""
        if not answers:
            return None

        answer_weights = {}
        for answer, weight in zip(answers, weights):
            if answer is not None:
                answer_str = str(answer)
                answer_weights[answer_str] = answer_weights.get(
                    answer_str, 0.0
                ) + float(weight)

        if not answer_weights:
            return None

        voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
        return voted_answer

    def process_trace(self, choice, trace_id):
        """处理单个trace并提取置信度"""
        text = choice.message.content
        tokens = [t.token for t in choice.logprobs.content]

        # 计算置信度
        confs = self.compute_confidence(
            [t.top_logprobs for t in choice.logprobs.content]
        )
        sliding_window = self.compute_least_grouped(confs, group_size=self.window_size)

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

    def inference(self, messages, verbose=False):
        """执行DeepConf推理"""
        if verbose:
            print("=" * 60)
            print("DEEPCONF INFERENCE")
            print("=" * 60)
            print(f"Model: {self.model_path}")
            print(f"Warmup traces: {self.warmup_traces}")
            print(f"Total budget: {self.total_budget}")
            print(f"Confidence percentile: {self.confidence_percentile}")
            print(f"Temperature: {self.temperature}")

        # ===========================
        # WARMUP阶段
        # ===========================
        if verbose:
            print(f"\n{'-'*40}")
            print("WARMUP PHASE")
            print(f"{'-'*40}")

        warmup_request_params = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "logprobs": True,
            "top_logprobs": 20,
            "n": self.warmup_traces,
            "extra_body": {"top_k": 0},
        }

        responses = self.client.chat.completions.create(**warmup_request_params)

        # 处理warmup traces
        warmup_trace_list = []
        min_confs = []

        for j in range(self.warmup_traces):
            trace_data = self.process_trace(responses.choices[j], j)
            warmup_trace_list.append(trace_data)
            min_confs.append(trace_data["min_conf"])

        # 计算置信度阈值
        conf_bar = float(np.percentile(min_confs, self.confidence_percentile))

        if verbose:
            print(f"Confidence bar (P{self.confidence_percentile}): {conf_bar:.4f}")
            print(f"Warmup traces generated: {len(warmup_trace_list)}")

        # ===========================
        # FINAL阶段
        # ===========================
        if verbose:
            print(f"\n{'-'*40}")
            print("FINAL PHASE (with early stopping)")
            print(f"{'-'*40}")

        real_gen = self.total_budget - self.warmup_traces

        final_request_params = {
            "model": self.model_path,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.95,
            "logprobs": True,
            "top_logprobs": 20,
            "n": real_gen,
            "extra_body": {
                "top_k": 0,
                "vllm_xargs": {
                    "enable_conf": True,
                    "window_size": self.window_size,
                    "threshold": conf_bar,
                },
            },
        }

        responses = self.client.chat.completions.create(**final_request_params)

        # 处理final traces
        final_traces = []
        for j in range(len(responses.choices)):
            trace_data = self.process_trace(
                responses.choices[j], self.warmup_traces + j
            )
            final_traces.append(trace_data)

        if verbose:
            print(
                f"Final traces generated: {len(final_traces)} (requested: {real_gen})"
            )

        # ===========================
        # 投票获得最终答案
        # ===========================
        if verbose:
            print(f"\n{'-'*40}")
            print("VOTING FOR FINAL ANSWER")
            print(f"{'-'*40}")

        all_traces = warmup_trace_list + final_traces

        # 收集高于阈值的答案用于投票
        voting_answers = []
        voting_weights = []

        # 添加高于阈值的warmup traces（使用置信度作为权重）
        for trace in warmup_trace_list:
            minx = trace["min_conf"]
            if minx >= conf_bar:
                answer = trace["text"]
                if answer is not None:
                    voting_answers.append(answer)
                    voting_weights.append(minx)

        # 添加高于阈值的final traces（使用权重1）
        for trace in final_traces:
            minx = trace["min_conf"]
            if minx >= conf_bar:
                answer = trace["text"]
                if answer is not None:
                    voting_answers.append(answer)
                    voting_weights.append(1.0)

        if verbose:
            print(f"Traces used for voting: {len(voting_answers)}")

        # 执行加权多数投票
        voted_answer = self.weighted_majority_vote(voting_answers, voting_weights)

        if verbose:
            print(f"Voted answer: {voted_answer}")

            # 显示投票细节
            if voting_answers:
                answer_counts = Counter(voting_answers)
                print(f"\nVoting breakdown:")
                for answer, count in answer_counts.most_common():
                    print(f"  {answer[:100]}...: {count} votes")

            print(f"\n{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"Confidence bar: {conf_bar:.4f}")
            print(f"Total traces generated: {len(all_traces)}")
            print(f"Traces used for voting: {len(voting_answers)}")
            print(f"Final answer: {voted_answer}")

        return voted_answer


# ===========================
# 核心函数
# ===========================
def request_model(
    query_list=None,
    answer_list0=None,
    time_str=None,
    location=None,
    all_p2q_res=False,
    temperature=TEMPERATURE,
    confidence_percentile=CONFIDENCE_PERCENTILE,
    window_size=WINDOW_SIZE,
    warmup_traces=WARMUP_TRACES,
    total_budget=TOTAL_BUDGET,
):
    """使用DeepConf处理查询及对话历史"""

    # 创建DeepConf实例
    deepconf = DeepConfInference(
        temperature=temperature,
        confidence_percentile=confidence_percentile,
        window_size=window_size,
        warmup_traces=warmup_traces,
        total_budget=total_budget,
    )

    answer_list = ["..."] + answer_list0
    system = PROMPT["system"].format(time=time_str, location=location)
    message_log = [{"role": "system", "content": system}]

    for query, answer in zip(query_list, answer_list):
        user = PROMPT["user"].format(
            ans=answer[:200] + answer[200:][-200:], query=query
        )
        message_log.append({"role": "user", "content": user})

        # 使用deepconf进行推理
        response = deepconf.inference(message_log, verbose=True)

        message_log.append({"role": "assistant", "content": response})

    if all_p2q_res:
        return [x["content"] for x in message_log if x["role"] == "assistant"]

    return message_log[-1]["content"]


def extract_response(response):
    """从模型响应中提取组件"""
    components = {"fix": "", "net": "", "dismantle": "", "source": "全网"}

    patterns = {
        "fix": r"<fix>(.*?)</fix>",
        "net": r"<net>(.*?)</net>",
        "dismantle": r"<dismantle>(.*?)</dismantle>",
        "source": r"<source>(.*?)</source>",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        components[key] = match.group(1).strip() if match else "error"

    return components


def main(
    csv_path,
    output_dir="results",
    temperature=TEMPERATURE,
    confidence_percentile=CONFIDENCE_PERCENTILE,
    window_size=WINDOW_SIZE,
    warmup_traces=WARMUP_TRACES,
    total_budget=TOTAL_BUDGET,
):
    """处理CSV文件并保存结果"""

    time_str = "2025年08月17日 9:30:46 星期六"
    location = "中国-北京市-北京市-朝阳区"
    results = []

    # 生成输出文件路径
    import os

    csv_filename = os.path.basename(csv_path)
    result_path = f"{output_dir}/deepconf_{csv_filename}"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV文件
    csv1 = pd.read_csv(csv_path)
    csv1 = csv1.fillna("")
    headers = list(csv1.columns)

    # 查找列索引
    query_idx = headers.index("query")

    # 查找历史对话的列索引
    round_cols = [
        ("第一轮问题", "第一轮回答"),
        ("第二轮问题", "第二轮回答"),
        ("第三轮问题", "第三轮回答"),
        ("第四轮问题", "第四轮回答"),
    ]

    # 验证所有需要的列都存在
    missing_cols = []
    for q_col, a_col in round_cols:
        if q_col not in headers:
            missing_cols.append(q_col)
        if a_col not in headers:
            missing_cols.append(a_col)

    if missing_cols:
        print(f"Warning: Missing columns in CSV: {missing_cols}")

    print(f"Processing {len(csv1)} rows from {csv_path}")
    print(f"Results will be saved to {result_path}")

    for idx, line in enumerate(csv1.values.tolist()):
        query0 = line[query_idx]
        query_list, answer_list = [], []

        if len(str(query0).strip()) < 1:
            continue

        # 构建对话历史 - 从第一轮到第四轮
        for q_col, a_col in round_cols:
            if q_col in headers and a_col in headers:
                q_idx = headers.index(q_col)
                a_idx = headers.index(a_col)

                query = str(line[q_idx]).strip() if q_idx < len(line) else ""
                answer = str(line[a_idx]).strip() if a_idx < len(line) else ""

                # 只有当问题不为空时才添加到历史对话
                if len(query) > 0:
                    query_list.append(query)
                    answer_list.append(answer)
                else:
                    break  # 如果某一轮问题为空，停止添加后续轮次

        # 将当前query添加到最后
        query_list.append(str(query0).strip())
        print(f"query_list: ")
        print(json.dumps(query_list, ensure_ascii=False))

        print(f"Processing row {idx + 1}/{len(csv1)}: {query0[:50]}...")

        try:
            # 调用模型
            response = request_model(
                query_list=query_list,
                answer_list0=answer_list,
                time_str=time_str,
                location=location,
                temperature=temperature,
                confidence_percentile=confidence_percentile,
                window_size=window_size,
                warmup_traces=warmup_traces,
                total_budget=total_budget,
            )

            # 解析响应
            components = extract_response(response)

            net = components["net"]
            dismantle = components["dismantle"]

            # 添加结果到行 - 只保存query、net、dismantle
            result_row = [query0, net, dismantle]
            results.append(result_row)

            print(f"  Net: {net}")
            print(f"  Dismantle: {dismantle[:100]}...")

        except Exception as e:
            print(f"Error processing row {idx + 1}: {str(e)}")
            # 添加错误结果 - 只保存query、net、dismantle
            result_row = [query0, "error", "error"]
            results.append(result_row)

    # 保存结果 - 只保存query、net、dismantle三列
    result_df = pd.DataFrame(results, columns=["query", "net", "dismantle"])
    result_df.to_csv(result_path, index=False)

    print(f"\nResults saved to: {result_path}")
    print(f"Processed {len(results)} rows successfully")


# ===========================
# 主入口
# ===========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepConf P2Q Processing")
    parser.add_argument(
        "--csv-path", "-i", type=str, required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=TEMPERATURE,
        help="Temperature for DeepConf (default: 0.9)",
    )
    parser.add_argument(
        "--confidence_percentile",
        "-c",
        type=int,
        default=CONFIDENCE_PERCENTILE,
        help="Confidence percentile for DeepConf (default: 10)",
    )
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        default=WINDOW_SIZE,
        help="Window size for DeepConf (default: 8)",
    )
    parser.add_argument(
        "--warmup_traces",
        "-u",
        type=int,
        default=WARMUP_TRACES,
        help="Warmup traces for DeepConf (default: 16)",
    )
    parser.add_argument(
        "--total_budget",
        "-b",
        type=int,
        default=TOTAL_BUDGET,
        help="Total budget for DeepConf (default: 64)",
    )
    args = parser.parse_args()

    # 运行主函数
    main(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        confidence_percentile=args.confidence_percentile,
        window_size=args.window_size,
        warmup_traces=args.warmup_traces,
        total_budget=args.total_budget,
    )
