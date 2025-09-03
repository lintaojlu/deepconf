import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import glob
import random
from typing import Dict, List, Tuple, Optional, Any
from dynasor.core.evaluator import math_equal


TRACE_DIRS = [YOURDIR1, YOURDIR2, ...]

# ============================================
# SECTION 1: Math Evaluation Functions
# ============================================


def parse_func(s):
    for f in [parse_latex, parse_expr, latex2sympy]:
        try:
            return f(s.replace("\\\\", "\\"))
        except:
            try:
                return f(s)
            except:
                pass
    return s

def quick_parse(text):
    """Quick parse to remove LaTeX text formatting"""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text

# ============================================
# SECTION 2: Answer Extraction
# ============================================

def extract_answer(text):
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None

# ============================================
# SECTION 3: Confidence Metrics Calculation
# ============================================

def calculate_confidence_stats(conf_list, tokens):
    """Calculate various confidence statistics"""
    if not conf_list:
        return {}

    assert len(conf_list) == len(tokens), "Confidence list and tokens must have same length"

    conf_array = np.array(conf_list)
    total_tokens = len(conf_array)

    stats = {
        'mean_confidence': np.mean(conf_array)
    }

    # First/Last N tokens
    for n in [2048]:
        if total_tokens >= n:
            stats[f'tail_{n}_mean_conf'] = np.mean(conf_array[-n:])
        else:
            stats[f'tail_{n}_mean_conf'] = np.mean(conf_array)

    # First/Last percentage
    for ratio in [0.1]:
        n_tokens = max(1, int(total_tokens * ratio))
        stats[f'tail_{ratio}_mean_conf'] = np.mean(conf_array[-n_tokens:])

    # Sliding window metrics
    window_sizes = [2048]
    bottom_percentages = [0.1, 0.5]

    for window_size in window_sizes:
        if total_tokens < window_size:
            stats[f'min_sliding_{window_size}_mean_conf'] = np.mean(conf_array)
            for percent in bottom_percentages:
                stats[f'bottom_{percent}_sliding_{window_size}_mean_conf'] = np.mean(conf_array)
        else:
            # Optimized sliding window
            cumsum = np.cumsum(conf_array)
            window_sums = cumsum[window_size-1:]
            window_sums[1:] -= cumsum[:-window_size]
            window_means = window_sums / window_size
            stats[f'min_sliding_{window_size}_mean_conf'] = np.min(window_means)

            sorted_means = np.sort(window_means)

            for percent in bottom_percentages:
                idx = int(len(sorted_means) * percent)
                stats[f'bottom_{percent}_sliding_{window_size}_mean_conf'] = sorted_means[:idx].mean()

    return stats

# ============================================
# SECTION 4: Voting Strategies
# ============================================

def majority_vote(traces):
    """Perform majority voting based on extracted answers"""
    if not traces:
        return None, None

    answer_counts = {}
    answer_to_parsed = {}

    for trace in traces:
        extracted_answer = trace.get('extracted_answer')
        parsed_answer = trace.get('parsed_answer')

        if extracted_answer is not None:
            answer_str = str(extracted_answer)
            answer_counts[answer_str] = answer_counts.get(answer_str, 0) + 1
            if answer_str not in answer_to_parsed:
                answer_to_parsed[answer_str] = parsed_answer

    if not answer_counts:
        return None, None

    voted_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x])
    voted_parsed = answer_to_parsed[voted_answer]

    return voted_answer, voted_parsed

def weighted_majority_vote(traces, weight_key='mean_confidence'):
    """Perform weighted majority voting"""
    if not traces:
        return None, None

    answer_weights = {}
    answer_to_parsed = {}

    for trace in traces:
        extracted_answer = trace.get('extracted_answer')
        parsed_answer = trace.get('parsed_answer')
        weight = trace.get(weight_key)

        if extracted_answer is not None and weight is not None:
            answer_str = str(extracted_answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
            if answer_str not in answer_to_parsed:
                answer_to_parsed[answer_str] = parsed_answer

    if not answer_weights:
        return None, None

    voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    voted_parsed = answer_to_parsed[voted_answer]

    return voted_answer, voted_parsed

def top_percent_vote(traces, weight_key='mean_confidence', top_percent=0.1, vote_strategy='majority'):
    """
    First filter top percent of traces by weight_key, then perform voting

    Args:
        traces: List of trace dictionaries
        weight_key: Key to use for filtering (e.g., 'mean_confidence')
        top_percent: Percentage of top traces to keep (e.g., 0.1 for top 10%)
        vote_strategy: 'majority' or 'weighted'

    Returns:
        voted_answer, voted_parsed
    """
    if not traces:
        return None, None

    # Filter traces that have the weight_key and valid answers
    valid_traces = [t for t in traces if weight_key in t and t.get('extracted_answer') is not None]

    if not valid_traces:
        return None, None

    # Sort traces by weight_key in descending order (higher is better)
    sorted_traces = sorted(valid_traces, key=lambda x: x[weight_key], reverse=True)

    # Select top percent
    n_top = max(1, int(len(sorted_traces) * top_percent))
    top_traces = sorted_traces[:n_top]

    # Apply voting strategy on filtered traces
    if vote_strategy == 'majority':
        return majority_vote(top_traces)
    elif vote_strategy == 'weighted':
        return weighted_majority_vote(top_traces, weight_key)
    else:
        raise ValueError(f"Unknown vote_strategy: {vote_strategy}")

# ============================================
# SECTION 5: JSONL Processing
# ============================================

def process_jsonl_file(file_path, ground_truth=None):
    """Process a single JSONL file and extract traces with metrics"""
    traces = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            data = json.loads(line)

            # Extract response and confidence data
            response = data.get('response', '')
            mean_confidences = data.get('mean_confidences', [])
            tokens = data.get('tokens', [])
            question_meta = data['question_meta']['original_question']

            # Extract answer
            extracted_answer = extract_answer(response)
            parsed_answer = parse_func(extracted_answer) if extracted_answer else None

            # Get ground truth
            if ground_truth is None:
                # Try to extract from question_meta
                for field in ['answer', 'solution', 'target']:
                    if field in question_meta:
                        ground_truth = str(question_meta[field]).strip()
                        break

            # Calculate confidence statistics
            conf_stats = calculate_confidence_stats(mean_confidences, tokens)

            # Check correctness
            is_correct = False
            if extracted_answer is not None and ground_truth is not None:
                is_correct = math_equal(extracted_answer, ground_truth)

            # Create trace entry
            trace = {
                'trace_id': data.get('trace_id', line_num),
                'extracted_answer': extracted_answer,
                'parsed_answer': parsed_answer,
                'is_correct': is_correct,
                'ground_truth': ground_truth,
                'response': response,
                **conf_stats
            }

            traces.append(trace)

        except Exception as e:
            print(f"Error processing line {line_num} in {file_path}: {e}")
            continue

    return traces

def process_multiple_jsonls(file_pattern, ground_truth_map=None):
    """Process multiple JSONL files matching a pattern"""
    files = glob.glob(file_pattern)
    all_data = defaultdict(list)

    for file_path in tqdm(files, desc="Processing JSONL files"):
        # Extract question ID from filename if possible
        filename = os.path.basename(file_path)
        question_id = None

        # Try to extract question ID (adjust pattern as needed)
        if '_processed.jsonl' in filename:
            try:
                question_id = int(filename.replace('_processed.jsonl', ''))
            except:
                question_id = filename
        else:
            question_id = filename

        # Get ground truth for this question
        ground_truth = None
        if ground_truth_map and question_id in ground_truth_map:
            ground_truth = ground_truth_map[question_id]

        # Process the file
        traces = process_jsonl_file(file_path, ground_truth)

        if traces:
            all_data[question_id] = traces

    return dict(all_data)

def process_multiple_dirs_jsonls(trace_dirs, file_pattern="*_processed.jsonl", ground_truth_map=None):
    """
    Process JSONL files from multiple directories and merge traces with same filename

    Args:
        trace_dirs: List of directory paths to search for JSONL files
        file_pattern: File pattern to match (e.g., "*_processed.jsonl")
        ground_truth_map: Optional dictionary mapping question IDs to ground truth answers

    Returns:
        Dictionary where keys are question IDs and values are lists of merged traces
    """
    all_data = defaultdict(list)

    # First, collect all unique filenames across all directories
    all_filenames = set()
    dir_file_mapping = defaultdict(list)  # Track which dirs have which files

    for trace_dir in trace_dirs:
        if not os.path.exists(trace_dir):
            print(f"Warning: Directory {trace_dir} does not exist, skipping...")
            continue

        pattern = os.path.join(trace_dir, file_pattern)
        files = glob.glob(pattern)
        print(f"Found {len(files)} files in {trace_dir}")

        for file_path in files:
            filename = os.path.basename(file_path)
            all_filenames.add(filename)
            dir_file_mapping[filename].append(trace_dir)

    print(f"Total unique filenames found: {len(all_filenames)}")

    # Process each unique filename across all directories
    for filename in tqdm(all_filenames, desc="Processing unique files"):
        # Extract question ID from filename
        question_id = None
        if '_processed.jsonl' in filename:
            try:
                question_id = int(filename.replace('_processed.jsonl', ''))
            except:
                question_id = filename
        else:
            question_id = filename

        # Get ground truth for this question
        ground_truth = None
        if ground_truth_map and question_id in ground_truth_map:
            ground_truth = ground_truth_map[question_id]

        # Collect traces from all directories for this filename
        merged_traces = []
        dirs_with_file = dir_file_mapping[filename]

        for trace_dir in dirs_with_file:
            file_path = os.path.join(trace_dir, filename)
            if os.path.exists(file_path):
                try:
                    traces = process_jsonl_file(file_path, ground_truth)

                    # Add directory info to each trace for identification
                    for i, trace in enumerate(traces):
                        trace['source_dir'] = trace_dir
                        trace['source_file'] = filename
                        # Create unique trace ID combining dir and original trace ID
                        original_trace_id = trace.get('trace_id', i)
                        trace['trace_id'] = f"{os.path.basename(trace_dir)}_{original_trace_id}"

                    merged_traces.extend(traces)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        if merged_traces:
            all_data[question_id] = merged_traces
            print(f"Question {question_id}: Merged {len(merged_traces)} traces from {len(dirs_with_file)} directories")

    return dict(all_data)

# ============================================
# SECTION 6: Analysis and Evaluation
# ============================================

def analyze_voting_performance(data, voting_sizes=[1, 2, 4, 8, 16, 32],
                              strategy='majority', weight_key='mean_confidence',
                              n_trials=1, seed=42, top_percent=None):
    """Analyze voting performance across different ensemble sizes"""

    random.seed(seed)
    np.random.seed(seed)

    results = {}
    for vote_size in voting_sizes:
        accuracies = []

        for trial in range(n_trials):
            correct = 0
            total = 0

            for question_id, traces in data.items():
                if len(traces) < vote_size:
                    continue

                # Sample traces
                sampled = random.sample(traces, vote_size)

                # Apply voting strategy
                if strategy == 'majority':
                    voted_answer, _ = majority_vote(sampled)
                elif strategy == 'weighted':
                    voted_answer, _ = weighted_majority_vote(sampled, weight_key)
                elif strategy == 'top_percent':
                    voted_answer, _ = top_percent_vote(sampled, weight_key, top_percent, 'majority')
                elif strategy == 'top_percent_weighted':
                    voted_answer, _ = top_percent_vote(sampled, weight_key, top_percent, 'weighted')
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                # Check correctness
                if voted_answer is not None and sampled[0]['ground_truth'] is not None:
                    ground_truth = sampled[0]['ground_truth']
                    if math_equal(voted_answer, ground_truth):
                        correct += 1
                    total += 1

            if total > 0:
                accuracies.append(correct / total)

        if accuracies:
            results[vote_size] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies)
            }

    return results

def analyze_top_percent_strategies(data, voting_sizes=[1, 2, 4, 8],
                                 weight_keys=['mean_confidence', 'tail_2048_mean_conf'],
                                 top_percents=[0.1, 0.2, 0.3, 0.5],
                                 n_trials=1, seed=42):
    """
    Comprehensive analysis of top percent filtering strategies

    Args:
        data: Processed trace data
        voting_sizes: List of ensemble sizes to test
        weight_keys: List of confidence metrics to use for filtering
        top_percents: List of top percentages to test (e.g., [0.1, 0.2] for top 10%, 20%)
        n_trials: Number of random trials per configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results for each configuration
    """

    results = {}

    # Test each combination of parameters
    for weight_key in weight_keys:
        for top_percent in top_percents:
            for vote_strategy in ['weighted']:
                strategy_name = f"top_{int(top_percent*100)}%_{vote_strategy}_{weight_key}"

                print(f"Testing {strategy_name}...")

                strategy = 'top_percent' if vote_strategy == 'majority' else 'top_percent_weighted'

                strategy_results = analyze_voting_performance(
                    data,
                    voting_sizes=voting_sizes,
                    strategy=strategy,
                    weight_key=weight_key,
                    n_trials=n_trials,
                    seed=seed,
                    top_percent=top_percent
                )

                results[strategy_name] = strategy_results

    return results

def analyze_directory_distribution(data):
    """Analyze the distribution of traces across source directories"""
    print("\n" + "="*60)
    print("DIRECTORY DISTRIBUTION ANALYSIS")
    print("="*60)

    dir_stats = defaultdict(lambda: {'total_traces': 0, 'questions': set()})

    for question_id, traces in data.items():
        for trace in traces:
            source_dir = trace.get('source_dir', 'unknown')
            dir_stats[source_dir]['total_traces'] += 1
            dir_stats[source_dir]['questions'].add(question_id)

    print(f"{'Directory':<50} {'Traces':<10} {'Questions':<10}")
    print("-" * 72)

    for dir_name, stats in dir_stats.items():
        short_name = os.path.basename(dir_name) if dir_name != 'unknown' else 'unknown'
        print(f"{short_name:<50} {stats['total_traces']:<10} {len(stats['questions']):<10}")

    return dir_stats

# ============================================
# MAIN EXECUTION CELL
# ============================================

def main_analysis_multi_dir(trace_dirs=None, file_pattern="*_processed.jsonl",
                           ground_truth_file=None, output_dir="./results"):
    """
    Main analysis function for multiple directories

    Args:
        trace_dirs: List of directory paths to search for JSONL files
        file_pattern: File pattern to match (e.g., "*_processed.jsonl")
        ground_truth_file: Optional pickle file with ground truth answers
        output_dir: Directory to save results
    """

    if trace_dirs is None:
        trace_dirs = TRACE_DIRS

    print("="*60)
    print("ENHANCED MULTI-DIRECTORY JSONL VOTING ANALYSIS")
    print("="*60)
    print(f"Processing {len(trace_dirs)} directories:")
    for i, dir_path in enumerate(trace_dirs):
        print(f"  {i+1}. {dir_path}")

    # Load ground truth if provided
    ground_truth_map = None
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'rb') as f:
            ground_truth_map = pickle.load(f)
        print(f"Loaded ground truth from {ground_truth_file}")

    # Process JSONL files from multiple directories
    print(f"\nProcessing files with pattern: {file_pattern}")
    data = process_multiple_dirs_jsonls(trace_dirs, file_pattern, ground_truth_map)

    print(f"Processed {len(data)} questions")
    total_traces = sum(len(traces) for traces in data.values())
    print(f"Total traces: {total_traces}")

    # Analyze directory distribution
    dir_stats = analyze_directory_distribution(data)

    # Calculate per-question statistics
    print("\n" + "="*60)
    print("PER-QUESTION STATISTICS")
    print("="*60)

    question_items = list(data.items())
    for q_id, traces in question_items[:5]:  # Show first 5 questions
        correct = sum(1 for t in traces if t['is_correct'])
        print(f"Question {q_id}: {correct}/{len(traces)} correct ({correct/len(traces):.2%})")
        if traces:
            mean_conf = np.mean([t.get('mean_confidence', 0) for t in traces if 'mean_confidence' in t])
            print(f"  Mean confidence: {mean_conf:.4f}")

            # Show directory breakdown
            dir_breakdown = defaultdict(int)
            for trace in traces:
                dir_name = os.path.basename(trace.get('source_dir', 'unknown'))
                dir_breakdown[dir_name] += 1
            print(f"  Directory breakdown: {dict(dir_breakdown)}")

    # Test baseline strategies
    print("\n" + "="*60)
    print("BASELINE VOTING STRATEGIES")
    print("="*60)

    baseline_strategies = [
        ('majority', 'Majority Vote', None),
        ('weighted', 'Weighted Vote (mean_conf)', 'mean_confidence'),
        ('weighted', 'Weighted Vote (tail_2048)', 'tail_2048_mean_conf'),

    ]

    voting_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    all_results = {}

    for strategy, name, weight_key in baseline_strategies:
        print(f"\n{name}:")
        results = analyze_voting_performance(
            data,
            voting_sizes=voting_sizes,
            strategy=strategy,
            weight_key=weight_key,
            n_trials=10
        )

        all_results[name] = results

        print(f"{'Size':<6} {'Accuracy':<12} {'Std Dev':<10}")
        print("-" * 30)
        for size in voting_sizes:
            if size in results:
                acc = results[size]['accuracy_mean']
                std = results[size]['accuracy_std']
                print(f"{size:<6} {acc:<12.4f} {std:<10.4f}")

    # Test top percent filtering strategies
    print("\n" + "="*60)
    print("TOP PERCENT FILTERING STRATEGIES")
    print("="*60)

    top_percent_results = analyze_top_percent_strategies(
        data,
        voting_sizes=voting_sizes,  # Use smaller sizes for top percent
        weight_keys=['mean_confidence', 'tail_2048_mean_conf', 'bottom_0.1_sliding_2048_mean_conf'],
        top_percents=[0.1, 0.9],
        n_trials=10
    )

    all_results.update(top_percent_results)

    # Display top percent results
    for strategy_name, strategy_results in top_percent_results.items():
        print(f"\n{strategy_name}:")
        print(f"{'Size':<6} {'Accuracy':<12} {'Std Dev':<10}")
        print("-" * 30)
        for size in voting_sizes:
            if size in strategy_results:
                acc = strategy_results[size]['accuracy_mean']
                std = strategy_results[size]['accuracy_std']
                print(f"{size:<6} {acc:<12.4f} {std:<10.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    data_path = os.path.join(output_dir, "processed_data_multi_dir.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✓ Saved processed data to {data_path}")

    # Save voting results
    results_path = os.path.join(output_dir, "voting_results_multi_dir.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"✓ Saved voting results to {results_path}")

    # Create comprehensive summary DataFrame
    summary_data = []
    for strategy_name, strategy_results in all_results.items():
        for size, metrics in strategy_results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Ensemble Size': size,
                'Accuracy': metrics['accuracy_mean'],
                'Std Dev': metrics['accuracy_std']
            })

    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "voting_summary_multi_dir.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Saved summary CSV to {csv_path}")

    # Find best performing strategies
    print("\n" + "="*60)
    print("BEST PERFORMING STRATEGIES")
    print("="*60)

    # Group by ensemble size and find best accuracy for each
    for size in voting_sizes:
        size_results = df_summary[df_summary['Ensemble Size'] == size]
        if not size_results.empty:
            best_row = size_results.loc[size_results['Accuracy'].idxmax()]
            print(f"Size {size}: {best_row['Strategy']} (Accuracy: {best_row['Accuracy']:.4f})")

    return data, all_results, df_summary, dir_stats

# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Example usage - modify these paths for your data

    # Use the predefined TRACE_DIRS or specify your own
    custom_dirs = None  # Set to your list of directories if different from TRACE_DIRS

    file_pattern = "*_processed.jsonl"  # Adjust this pattern
    ground_truth_file = None  # Optional: "./ground_truth.pkl"
    output_directory = "./voting_results_multi_dir"

    # Run the analysis
    data, results, summary_df, dir_stats = main_analysis_multi_dir(
        trace_dirs=custom_dirs,  # Will use TRACE_DIRS if None
        file_pattern=file_pattern,
        ground_truth_file=ground_truth_file,
        output_dir=output_directory
    )

    # Display summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Show some example merged data
    print("\n" + "="*60)
    print("EXAMPLE MERGED DATA")
    print("="*60)

    for q_id, traces in list(data.items())[:2]:  # Show first 2 questions
        print(f"\nQuestion {q_id} ({len(traces)} traces):")
        for trace in traces[:3]:  # Show first 3 traces
            source_dir = os.path.basename(trace.get('source_dir', 'unknown'))
            correct = trace.get('is_correct', False)
            conf = trace.get('mean_confidence', 0)
            answer = trace.get('extracted_answer', 'None')
            print(f"  {source_dir}: {answer} (correct: {correct}, conf: {conf:.4f})") 