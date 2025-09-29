import openai
import numpy as np
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===========================
# Configuration
# ===========================
MODEL_PATH = "qsite_0.6b_v7.49"
MAX_TOKENS = 12800
PORT = 8000
TEMPERATURE = 0.9
TOP_P = 0.8
TOP_K = 20

# Inference parameters
WARMUP_TRACES = 8
TOTAL_BUDGET = 16
CONFIDENCE_PERCENTILE = 50
WINDOW_SIZE = 32

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)


# ===========================
# Confidence Calculation
# ===========================
def _compute_confidence(logprobs):
    """Compute confidence score from logprobs."""
    confs = []
    for lp in logprobs:
        confs.append(round(-sum([entry.logprob for entry in lp]) / len(lp), 3))
    return confs


def _compute_least_grouped(confs, group_size=WINDOW_SIZE):
    """Compute sliding window mean confidence with specified group size."""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]

    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))

    return sliding_means


# ===========================
# Trace Processing
# ===========================
def _process_trace(choice, trace_id):
    """Process a single trace and extract confidence"""
    text = choice.message.content
    confs = _compute_confidence([t.top_logprobs for t in choice.logprobs.content])
    sliding_window = _compute_least_grouped(confs, group_size=WINDOW_SIZE)

    trace_data = {
        "trace_id": trace_id,
        "text": text,
        "confs": confs,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
    }

    return trace_data


def _extract_query_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract a reasonable query string from messages (last 'user' content or empty)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    # Fallback to last message content
    return str(messages[-1].get("content", "")) if messages else ""


def _run_single_deepconf(messages: List[Dict[str, Any]],
                         warmup_traces: int,
                         total_budget: int,
                         confidence_percentile: int) -> Dict[str, Any]:
    """Run deepconf inference for a single messages list and aggregate unique answers."""
    query = _extract_query_from_messages(messages)
    logger.debug(f"Starting inference for query: {query[:100]}...")
    
    # Initialize client
    client = openai.OpenAI(
        api_key="None",
        base_url=f"http://localhost:{PORT}/v1",
        timeout=120
    )

    # WARMUP
    warmup_request_params = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "logprobs": True,
        "top_logprobs": 20,
        "n": warmup_traces,
        "extra_body": {"top_k": TOP_K},
    }

    try:
        responses = client.chat.completions.create(**warmup_request_params)
        logger.debug(f"Warmup phase completed, got {len(responses.choices)} traces")
    except Exception as e:
        logger.error(f"Warmup request failed: {e}")
        return {"query": query, "answer_list": []}

    warmup_trace_list = []
    min_confs = []
    for j in range(warmup_traces):
        trace_data = _process_trace(responses.choices[j], j)
        warmup_trace_list.append(trace_data)
        min_confs.append(trace_data["min_conf"])

    # Confidence bar
    conf_bar = float(np.percentile(min_confs, confidence_percentile)) if min_confs else 0.0
    logger.debug(f"Confidence bar set to {conf_bar:.4f} (P{confidence_percentile})")

    # FINAL with early stopping enabled via threshold
    real_gen = max(0, total_budget - warmup_traces)
    if real_gen > 0:
        final_request_params = {
            "model": MODEL_PATH,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "logprobs": True,
            "top_logprobs": 20,
            "n": real_gen,
            "extra_body": {
                "top_k": TOP_K,
                "vllm_xargs": {
                    'enable_conf': True,
                    'window_size': WINDOW_SIZE,
                    'threshold': conf_bar
                }
            }
        }
        try:
            responses = client.chat.completions.create(**final_request_params)
            logger.debug(f"Final phase completed, got {len(responses.choices)} traces (requested {real_gen})")
            final_traces = []
            for j in range(len(responses.choices)):
                trace_data = _process_trace(responses.choices[j], warmup_traces + j)
                final_traces.append(trace_data)
        except Exception as e:
            logger.error(f"Final request failed: {e}")
            final_traces = []
    else:
        final_traces = []
        logger.debug("Skipping final phase (no remaining budget)")

    # Aggregate answers (unique, preserve first-seen order)
    unique_answers = []
    seen = set()

    def _maybe_add(answer: str):
        if answer is None:
            return
        s = str(answer)
        if s not in seen:
            seen.add(s)
            unique_answers.append(s)

    for t in warmup_trace_list:
        if t["min_conf"] >= conf_bar:
            _maybe_add(t["text"])

    for t in final_traces:
        if t["min_conf"] >= conf_bar:
            _maybe_add(t["text"])

    logger.debug(f"Query completed: {len(unique_answers)} unique answers aggregated")
    return {
        "query": query,
        "answer_list": unique_answers,
    }


# ===========================
# Public API
# ===========================
def qsite_infer_deepconf(messages_list: List[List[Dict[str, Any]]],
                         warmup_traces: int = WARMUP_TRACES,
                         total_budget: int = TOTAL_BUDGET,
                         confidence_percentile: int = CONFIDENCE_PERCENTILE,
                         max_workers: int = None) -> List[Dict[str, Any]]:
    """
    Run deepconf inference concurrently for multiple messages and aggregate unique answers.

    Args:
        messages_list: List of message arrays, each suitable for Chat Completions API.
        warmup_traces: Number of warmup traces to generate per query.
        total_budget: Total budget (warmup + final) per query.
        confidence_percentile: Percentile for confidence threshold per query.

    Returns:
        results: List[dict] aligned with input order. Each dict has keys:
            - query: the extracted user query string
            - answer_list: unique aggregated answers (no voting)
    """

    logger.info(f"Starting batch inference for {len(messages_list)} queries with max_workers={max_workers}")
    
    # High concurrency while preserving order
    results: List[Dict[str, Any]] = [None] * len(messages_list)
    completed_count = 0

    def _task(idx: int, msgs: List[Dict[str, Any]]):
        return idx, _run_single_deepconf(msgs, warmup_traces, total_budget, confidence_percentile)

    max_workers = max(1, min(32, len(messages_list))) if max_workers is None else max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, i, msgs) for i, msgs in enumerate(messages_list)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(messages_list):
                logger.info(f"Progress: {completed_count}/{len(messages_list)} queries completed")

    logger.info("Batch inference completed")
    return results


def main():
    import argparse
    import json
    import pandas as pd
    
    global MODEL_PATH, PORT, TEMPERATURE, TOP_P, TOP_K

    parser = argparse.ArgumentParser(description='Batch deepconf inference with order-preserving merge')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='Path to input file; each line is a JSON array of messages')
    parser.add_argument('--output-file', '-o', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--warmup-traces', '-w', type=int, default=WARMUP_TRACES, help=f'Number of warmup traces (default: {WARMUP_TRACES})')
    parser.add_argument('--total-budget', '-t', type=int, default=TOTAL_BUDGET, help=f'Total budget (default: {TOTAL_BUDGET})')
    parser.add_argument('--confidence-percentile', '-c', type=int, default=CONFIDENCE_PERCENTILE, help=f'Confidence percentile (default: {CONFIDENCE_PERCENTILE})')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Model path/name')
    parser.add_argument('--port', type=int, default=PORT, help='Serving port of OpenAI-compatible API')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=TOP_P, help='Nucleus sampling top_p')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Top-k sampling')
    parser.add_argument('--max-workers', type=int, default=None, help='Max worker threads for concurrency')
    parser.add_argument('--limit', type=int, default=None, help='Only process first N lines for quick check')
    args = parser.parse_args()

    MODEL_PATH = args.model
    PORT = args.port
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    TOP_K = args.top_k

    logger.info(f"Configuration: model={MODEL_PATH}, port={PORT}, temp={TEMPERATURE}, "
                f"top_p={TOP_P}, top_k={TOP_K}, warmup={args.warmup_traces}, "
                f"budget={args.total_budget}, conf_pct={args.confidence_percentile}")

    batch_size = 10000
    total_processed = 0
    batch_count = 0
    is_first_batch = True

    with open(args.input_file, 'r', encoding='utf-8') as f:
        while True:
            # Read batch
            messages_batch = []
            lines_read = 0
            
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    msgs = data["messages"]
                    
                    if not isinstance(msgs, list):
                        continue
                    messages_batch.append(msgs)
                    lines_read += 1
                    
                    # Check limits
                    if len(messages_batch) >= batch_size:
                        break
                    if args.limit is not None and total_processed + len(messages_batch) >= args.limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to parse line {total_processed + lines_read + 1}: {e}")
                    continue
            
            # No more data
            if not messages_batch:
                break
                
            batch_count += 1
            logger.info(f"Processing batch {batch_count}: {len(messages_batch)} messages")
            
            # Process batch
            results = qsite_infer_deepconf(
                messages_list=messages_batch,
                warmup_traces=args.warmup_traces,
                total_budget=args.total_budget,
                confidence_percentile=args.confidence_percentile,
                max_workers=args.max_workers,
            )
            
            # Convert to DataFrame
            records = []
            for item in results:
                records.append({
                    'query': item.get('query', ''),
                    'answer_list': json.dumps(item.get('answer_list', []), ensure_ascii=False),
                })
            df = pd.DataFrame.from_records(records)
            
            # Save batch results
            if is_first_batch:
                df.to_csv(args.output_file, index=False, encoding='utf-8', mode='w')
                is_first_batch = False
                logger.info(f"Created CSV file: {args.output_file}")
            else:
                df.to_csv(args.output_file, index=False, encoding='utf-8', mode='a', header=False)
                
            total_processed += len(messages_batch)
            logger.info(f"Batch {batch_count} completed. Total processed: {total_processed}")
            
            # Check if we've hit the limit
            if args.limit is not None and total_processed >= args.limit:
                break

    if total_processed == 0:
        logger.error("No valid messages found in input file!")
        return

    logger.info(f"Successfully processed {total_processed} messages in {batch_count} batches")


if __name__ == '__main__':
    main()


