import openai
import json
from tqdm import tqdm
import time
import os
import requests
from datetime import datetime
from transformers import AutoTokenizer
import concurrent.futures
import threading
from functools import partial

# ===========================
# Model Configurations
# ===========================

MODEL_CONFIGS = {
    "Qwen/Qwen3-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 32000,
        "template": "qwen3"
    },
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 0,
        "max_tokens": 64000,
        "template": "dpsk_qwen_0528"
    },
    "openai/gpt-oss-20b": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "max_tokens": 14000,
        "template": "gpt"
    },
    # Add more model configurations as needed
}

# ===========================
# Main Configuration
# ===========================

# Select your model
MODEL_NAME = "Qwen/Qwen3-8B"  # Change this to your desired model
SAMPLES_PER_QUESTION = 4  # Number of traces to generate per question
DATASET_FILE = "aime25.jsonl"  # Input dataset file
REASONING_EFFORT = "high"  # For GPT models: low, medium, high

# Parallel processing configuration
MAX_WORKERS = 8  # Maximum number of concurrent workers (adjust based on your server capacity)
MAX_WORKERS_PER_QUESTION = 4  # Maximum workers for traces within a single question

# Get model-specific config
model_config = MODEL_CONFIGS.get(MODEL_NAME)

# General Configuration
CONFIG = {
    "model_path": MODEL_NAME,
    "server_port": 8000,
    "temperature": model_config["temperature"],
    "top_p": model_config["top_p"],
    "top_k": model_config["top_k"],
    "max_tokens": model_config["max_tokens"],
    "template": model_config["template"],
    "reasoning_effort": REASONING_EFFORT,

    # Dataset and sampling configuration
    "dataset": DATASET_FILE,  # Input dataset file
    "max_samples_per_question": SAMPLES_PER_QUESTION,  # Number of traces per question
    "output_dir": f"output_{MODEL_NAME.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",

    # Parallel processing configuration
    "max_workers": MAX_WORKERS,
    "max_workers_per_question": MAX_WORKERS_PER_QUESTION,
}

# Thread-safe file writing lock
file_lock = threading.Lock()

# ===========================
# Initialize OpenAI Client
# ===========================

# Note: Make sure vLLM server is already running on the specified port
# Example command to start vLLM server:
# vllm serve MODEL_NAME --port 8000 -tp 1 --gpu-memory-utilization 0.7 --enable-prefix-caching

print(f"Connecting to vLLM server...")
print(f"Model: {CONFIG['model_path']}")
print(f"Server URL: http://localhost:{CONFIG['server_port']}/v1")
print(f"Max concurrent workers: {CONFIG['max_workers']}")
print(f"Max workers per question: {CONFIG['max_workers_per_question']}")

# Initialize OpenAI client
client = openai.OpenAI(
    api_key="None",
    base_url=f"http://localhost:{CONFIG['server_port']}/v1",
    timeout=None
)

# Initialize tokenizer for GPT models
if CONFIG['template'] == 'gpt':
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'])
else:
    tokenizer = None

# Test connection
try:
    response = requests.get(
        f"http://localhost:{CONFIG['server_port']}/v1/models",
        headers={"Authorization": "Bearer None"},
    )
    if response.status_code == 200:
        print("✅ Successfully connected to vLLM server")
    else:
        print(f"⚠️ Server returned status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"❌ Failed to connect to vLLM server: {e}")
    print("Please ensure vLLM server is running on the specified port")

# ===========================
# Core Processing Functions
# ===========================

def get_gpt_token_probabilities(messages, max_tokens=50):
    """
    Function to get token probabilities for GPT models using completions API
    """
    response = client.completions.create(
        model=CONFIG['model_path'],
        prompt=messages,
        max_tokens=max_tokens,
        temperature=CONFIG['temperature'],
        top_p=CONFIG['top_p'],
        logprobs=20,
        extra_body={
            "top_k": CONFIG['top_k']
        },
    )

    # Extract generated text
    generated_text = response.choices[0].text

    # Extract token probabilities
    token_probs = []
    mean_confs = []
    tokens = []
    log_probs = []

    if response.choices[0].logprobs and response.choices[0].logprobs.tokens:
        for i, token_data in enumerate(response.choices[0].logprobs.tokens):
            step_probs = {
                "s": i,  # step -> s
                "t": response.choices[0].logprobs.tokens[i],  # generated_token -> t
                "lp": round(response.choices[0].logprobs.token_logprobs[i], 2),  # logprob of generated token
                "a": []  # top_20_tokens -> a (alternatives)
            }

            # Add only top 5 alternatives to save space
            if response.choices[0].logprobs.top_logprobs:
                for tok, value in response.choices[0].logprobs.top_logprobs[i].items():  # Only top 5
                    step_probs["a"].append([
                        tok,
                        round(value, 2)
                    ])  # Use array instead of dict

            token_probs.append(step_probs)
            if step_probs['a']:
                mean_confs.append(round(-sum(p[1] for p in step_probs['a']) / len(step_probs['a']), 2))
            else:
                mean_confs.append(0)
            tokens.append(response.choices[0].logprobs.tokens[i])
            log_probs.append(round(response.choices[0].logprobs.token_logprobs[i], 2))

    return {
        "text": generated_text,
        "probs": token_probs,  # token_probabilities -> probs,
        "mean_confidences": mean_confs,  # mean_confidences -> mean_confs
        "tokens": tokens,
        "log_probs": log_probs  # log_probs -> log_probs
    }

def get_token_probabilities(prompt, messages):
    """Get token probabilities from the vLLM server using chat completions API."""
    response = client.chat.completions.create(
        model=CONFIG['model_path'],
        messages=messages,
        max_tokens=CONFIG['max_tokens'],
        temperature=CONFIG['temperature'],
        top_p=CONFIG['top_p'],
        logprobs=True,
        top_logprobs=20,
        extra_body={"top_k": CONFIG['top_k']},
    )

    generated_text = response.choices[0].message.content
    token_probs = []
    mean_confs = []
    tokens = []
    log_probs = []

    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        for i, token_data in enumerate(response.choices[0].logprobs.content):
            step_probs = {
                "s": i,
                "t": token_data.token,
                "lp": round(token_data.logprob, 2),
                "a": []
            }

            if token_data.top_logprobs:
                for logprob_data in token_data.top_logprobs[:5]:  # Top 5 alternatives
                    step_probs["a"].append([
                        logprob_data.token,
                        round(logprob_data.logprob, 2)
                    ])

            token_probs.append(step_probs)
            if step_probs['a']:
                mean_confs.append(round(-sum(p[1] for p in step_probs['a']) / len(step_probs['a']), 2))
            else:
                mean_confs.append(0)
            tokens.append(token_data.token)
            log_probs.append(round(token_data.logprob, 2))

    return {
        "text": generated_text,
        "probs": token_probs,
        "mean_confidences": mean_confs,
        "tokens": tokens,
        "log_probs": log_probs
    }

def prepare_messages(prompt, template):
    """Prepare messages based on template."""
    if template == "dpsk_qwen_0528":
        return [
            {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
            {"role": "user", "content": prompt}
        ]
    elif template == 'qwen3':
        return [
            {"role": "user", "content": prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
        ]
    elif template == 'gpt':
        # For GPT models, we'll prepare a simple string message first
        return prompt + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    else:
        return [{"role": "user", "content": prompt}]

def generate_single_trace(question_meta, trace_idx, output_dir):
    """Generate a single trace for a question. This function will be run in parallel."""
    try:
        prompt = question_meta["prompt"]
        q_idx = question_meta["question_id"]

        messages = prepare_messages(prompt, CONFIG['template'])

        # Handle GPT models differently
        if CONFIG['template'] == 'gpt':
            # Apply chat template with reasoning effort for GPT models
            if tokenizer:
                formatted_messages = tokenizer.apply_chat_template(
                    conversation=[
                        {"role": "user", "content": messages}
                    ],
                    add_generation_prompt=True,
                    reasoning_effort=CONFIG['reasoning_effort'],
                    tokenize=False,
                )
                result = get_gpt_token_probabilities(messages=formatted_messages, max_tokens=CONFIG['max_tokens'])
            else:
                # Fallback if tokenizer is not available
                result = get_gpt_token_probabilities(messages=messages, max_tokens=CONFIG['max_tokens'])
        else:
            # Use chat completions for other models
            result = get_token_probabilities(prompt, messages)

        # Prepare trace data
        trace_data_processed = {
            "question_meta": question_meta,
            "trace_id": trace_idx,
            "response": result["text"],
            "tokens": result["tokens"],
            "mean_confidences": result["mean_confidences"],
            "log_probs": result["log_probs"],
            "messages": messages,
        }

        # Thread-safe file writing
        processed_file = os.path.join(output_dir, f"{q_idx}_processed.jsonl")
        with file_lock:
            with open(processed_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data_processed, ensure_ascii=False) + "\n")

        return True, None

    except Exception as e:
        return False, f"Error in question {question_meta['question_id']}, trace {trace_idx}: {e}"

def process_question_parallel(question, q_idx, output_dir):
    """Process a single question and generate multiple traces in parallel."""
    prompt = question.get("problem", question.get("question", question.get("prompt", "")))

    if not prompt:
        print(f"Warning: No prompt found in question {q_idx}")
        return 0

    question_meta = {
        "question_id": q_idx,
        "original_question": question,
        "prompt": prompt,
    }

    # Generate traces in parallel
    completed_traces = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers_per_question']) as executor:
        # Submit all trace generation tasks
        future_to_trace = {
            executor.submit(generate_single_trace, question_meta, trace_idx, output_dir): trace_idx
            for trace_idx in range(CONFIG['max_samples_per_question'])
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_trace):
            trace_idx = future_to_trace[future]
            try:
                success, error_msg = future.result()
                if success:
                    completed_traces += 1
                else:
                    print(f"Error: {error_msg}")
            except Exception as e:
                print(f"Exception in trace {trace_idx}: {e}")

    return completed_traces

def process_single_question_wrapper(args):
    """Wrapper function for processing a single question (needed for parallel execution)."""
    question, q_idx, output_dir = args
    return q_idx, process_question_parallel(question, q_idx, output_dir)

def process_dataset_parallel(dataset_file, output_dir):
    """Process entire dataset with parallel processing."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Load dataset
    questions = []
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line.strip()))
        print(f"Loaded {len(questions)} questions from {dataset_file}")
    except FileNotFoundError:
        print(f"Error: {dataset_file} not found!")
        return None

    # Process questions in parallel
    all_results = []
    total_traces = 0

    # Prepare arguments for parallel processing
    question_args = [(question, q_idx, output_dir) for q_idx, question in enumerate(questions)]

    print(f"Processing {len(questions)} questions with up to {CONFIG['max_workers']} parallel workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        # Submit all question processing tasks
        future_to_question = {
            executor.submit(process_single_question_wrapper, args): args[1]
            for args in question_args
        }

        # Use tqdm to track progress
        with tqdm(total=len(questions), desc="Processing questions") as pbar:
            for future in concurrent.futures.as_completed(future_to_question):
                q_idx = future_to_question[future]
                try:
                    result_q_idx, traces_completed = future.result()
                    total_traces += traces_completed
                    all_results.append({
                        "question_id": result_q_idx,
                        "total_traces": traces_completed,
                        "file_path": os.path.join(output_dir, f"{result_q_idx}.jsonl")
                    })
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed_traces': total_traces,
                        'avg_traces': f"{total_traces / len(all_results):.1f}" if all_results else "0"
                    })
                except Exception as e:
                    print(f"Exception processing question {q_idx}: {e}")
                    pbar.update(1)

    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    summary = {
        "model": CONFIG['model_path'],
        "model_config": model_config,
        "dataset_file": dataset_file,
        "total_questions": len(questions),
        "completed_questions": len(all_results),
        "total_traces": total_traces,
        "average_traces_per_question": total_traces / len(all_results) if all_results else 0,
        "output_directory": output_dir,
        "timestamp": datetime.now().isoformat(),
        "reasoning_effort": CONFIG.get('reasoning_effort', 'N/A'),
        "parallel_config": {
            "max_workers": CONFIG['max_workers'],
            "max_workers_per_question": CONFIG['max_workers_per_question']
        }
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Completed! Generated {total_traces} total traces")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Summary: {summary_file}")
    print(f"📈 Average traces per question: {total_traces / len(all_results):.1f}")

    return output_dir

def check_results(output_dir):
    """Check and display results from output directory."""
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} not found!")
        return

    # Load summary
    summary_file = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"\nSummary:")
        print(f"  Model: {summary['model']}")
        print(f"  Total questions: {summary['total_questions']}")
        print(f"  Total traces: {summary['total_traces']}")
        print(f"  Average traces per question: {summary['average_traces_per_question']:.1f}")
        print(f"  Reasoning effort: {summary.get('reasoning_effort', 'N/A')}")
        if 'parallel_config' in summary:
            print(f"  Max workers: {summary['parallel_config']['max_workers']}")
            print(f"  Max workers per question: {summary['parallel_config']['max_workers_per_question']}")

    # Check individual files
    question_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    question_files.sort(key=lambda x: int(x.split('.')[0].split('_')[0]) if x.split('.')[0].split('_')[0].isdigit() else float('inf'))

    print(f"\nFound {len(question_files)} question files")

    # Show sample results
    for filename in question_files[:3]:
        if '_processed' in filename:
            continue

        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if lines:
                first_trace = json.loads(lines[0].strip())
                print(f"\n{filename}:")
                print(f"  Traces: {len(lines)}")
                print(f"  First response preview: {first_trace['response'][:150]}...")

# ===========================
# Performance Monitoring
# ===========================

def monitor_performance():
    """Monitor and suggest optimal worker configuration."""
    import psutil

    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    print(f"\n🔧 System Information:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Total memory: {memory_gb:.1f} GB")

    # Suggest optimal configuration
    suggested_workers = min(cpu_count * 2, 16)  # Generally good for I/O bound tasks
    suggested_workers_per_q = min(4, suggested_workers // 2)

    print(f"\n💡 Suggested Configuration:")
    print(f"   MAX_WORKERS: {suggested_workers}")
    print(f"   MAX_WORKERS_PER_QUESTION: {suggested_workers_per_q}")

    if CONFIG['max_workers'] > suggested_workers:
        print(f"⚠️  Current MAX_WORKERS ({CONFIG['max_workers']}) might be too high for your system")

    return suggested_workers, suggested_workers_per_q

# ===========================
# Main Execution
# ===========================

# Monitor system performance
monitor_performance()

print(f"\n🚀 Starting parallel processing:")
print(f"   Model: {CONFIG['model_path']}")
print(f"   Template: {CONFIG['template']}")
print(f"   Dataset: {CONFIG['dataset']}")
print(f"   Traces per question: {CONFIG['max_samples_per_question']}")
print(f"   Max workers: {CONFIG['max_workers']}")
print(f"   Max workers per question: {CONFIG['max_workers_per_question']}")
print(f"   Output directory: {CONFIG['output_dir']}")
if CONFIG['template'] == 'gpt':
    print(f"   Reasoning effort: {CONFIG['reasoning_effort']}")

start_time = time.time()

# Process the dataset with parallel processing
output_dir = process_dataset_parallel(
    dataset_file=CONFIG['dataset'],
    output_dir=CONFIG['output_dir']
)

end_time = time.time()
processing_time = end_time - start_time

# Check results
if output_dir:
    print("\n" + "="*50)
    print("Checking generated results...")
    check_results(output_dir)

print(f"\n⚡ Processing completed in {processing_time:.2f} seconds!")
print(f"🚀 Speed improvement with parallel processing!")
print("Note: vLLM server is still running. Stop it manually if needed.")