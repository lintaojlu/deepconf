#!/usr/bin/env bash

# Get the directory containing this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Create output directory if it doesn't exist
OUTPUT_DIR="${SCRIPT_DIR}/../benchmark-output"
mkdir -p "${OUTPUT_DIR}"

# Run the token deprivation experiment
set -x
python "${SCRIPT_DIR}/run.py" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset "math500" \
    --step 32 \
    --max-tokens 16384 \
    --start 0 \
    --end 10 \
    --output "${OUTPUT_DIR}/math500_step32_max16384_trials10" \
    --probe-tokens 32 \
    --probe "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{" \
    "$@"
set +x