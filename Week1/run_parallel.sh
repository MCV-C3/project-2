#!/bin/bash

# Parallel Grid Search Runner
# Usage: ./run_parallel.sh [n_splits]
# Example: ./run_parallel.sh 4  # Runs experiments in 4 parallel processes

set -e  # Exit on error

# Use python3 if python is not available
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Default to 4 parallel processes (safe default)
N_SPLITS=${1:-4}

echo "============================================"
echo "Parallel BoVW Grid Search Runner"
echo "============================================"
echo "Using Python: $PYTHON_CMD"

# Step 1: Count total configurations
echo "Counting total configurations..."
TOTAL_CONFIGS=$($PYTHON_CMD main.py --count-configs | grep "Total number of configurations:" | awk '{print $NF}')

if [ -z "$TOTAL_CONFIGS" ]; then
    echo "Error: Could not determine total number of configurations"
    exit 1
fi

echo "Total configurations: $TOTAL_CONFIGS"
echo "Number of parallel processes: $N_SPLITS"

# Calculate configurations per split
CONFIGS_PER_SPLIT=$(( (TOTAL_CONFIGS + N_SPLITS - 1) / N_SPLITS ))
echo "Configurations per process: ~$CONFIGS_PER_SPLIT"

# Create logs directory
LOGS_DIR="logs/parallel_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"
echo "Logs will be saved to: $LOGS_DIR"

# Function to run a subset of configurations
run_split() {
    local split_id=$1
    local start_idx=$2
    local end_idx=$3
    local log_file="$LOGS_DIR/split_${split_id}.log"

    echo "  [Split $split_id] Running configs $start_idx to $end_idx"

    # Build the range string
    if [ $start_idx -eq $end_idx ]; then
        range="$start_idx"
    else
        range="$start_idx-$end_idx"
    fi

    # Run the experiment and log output
    $PYTHON_CMD main.py --run "$range" > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "  [Split $split_id] ✓ Completed successfully"
    else
        echo "  [Split $split_id] ✗ Failed (check $log_file)"
    fi
}

# Launch parallel processes
echo ""
echo "Starting parallel execution..."
echo "============================================"

pids=()
for i in $(seq 0 $((N_SPLITS - 1))); do
    start_idx=$((i * CONFIGS_PER_SPLIT))
    end_idx=$((start_idx + CONFIGS_PER_SPLIT - 1))

    # Don't exceed total configs
    if [ $end_idx -ge $TOTAL_CONFIGS ]; then
        end_idx=$((TOTAL_CONFIGS - 1))
    fi

    # Skip if start_idx is beyond total configs
    if [ $start_idx -ge $TOTAL_CONFIGS ]; then
        break
    fi

    # Run in background
    run_split "$i" "$start_idx" "$end_idx" &
    pids+=($!)
done

# Wait for all processes to complete
echo ""
echo "Waiting for all processes to complete..."
echo "You can monitor progress with: tail -f $LOGS_DIR/split_*.log"
echo ""

for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "============================================"
echo "All parallel processes completed!"
echo "============================================"
echo ""
echo "Check individual logs in: $LOGS_DIR/"
echo "You can view results with: grep -r 'Test set accuracy' $LOGS_DIR/"
echo "Or best configs with: grep -r 'BEST CONFIGURATION' $LOGS_DIR/"
