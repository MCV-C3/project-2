#!/bin/bash

# OPTIONAL 1: Parallel Feature Count Experiment Runner
# Usage: ./run_parallel_nfeatures.sh [n_splits]
# Example: ./run_parallel_nfeatures.sh 4  # Run on 4 cores

set -e

# Use python3 if python is not available
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Default to 4 parallel processes
N_SPLITS=${1:-4}

echo "============================================"
echo "OPTIONAL 1: Feature Count Experiment"
echo "Parallel Runner"
echo "============================================"
echo "Using Python: $PYTHON_CMD"
echo "Number of parallel processes: $N_SPLITS"
echo "Total configurations: 20 (10 SIFT + 10 AKAZE)"
echo ""

# Create logs directory
LOGS_DIR="logs/nfeatures_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# Function to run a subset of configurations
run_split() {
    local split_id=$1
    local start_idx=$2
    local end_idx=$3
    local descriptor=$4
    local log_file="$LOGS_DIR/${descriptor}_${start_idx}-${end_idx}.log"

    echo "  [Split $split_id] Running $descriptor configs $start_idx-$end_idx"

    # Build the range string
    if [ $start_idx -eq $end_idx ]; then
        range="$start_idx"
    else
        range="$start_idx-$end_idx"
    fi

    # Run the experiment and log output
    $PYTHON_CMD main_nfeatures.py --run "$range" > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "  [Split $split_id] ✓ $descriptor configs $start_idx-$end_idx completed"
    else
        echo "  [Split $split_id] ✗ $descriptor configs $start_idx-$end_idx failed (check $log_file)"
    fi
}

# Strategy: Split 20 configs across N processes
if [ $N_SPLITS -eq 4 ]; then
    # Optimal for 4 cores: 2 SIFT groups, 2 AKAZE groups
    echo "Running in 4 parallel processes..."
    echo "  - SIFT 0-4 (5 configs)"
    echo "  - SIFT 5-9 (5 configs)"
    echo "  - AKAZE 10-14 (5 configs)"
    echo "  - AKAZE 15-19 (5 configs)"
    echo ""

    run_split 0 0 4 "SIFT" &
    run_split 1 5 9 "SIFT" &
    run_split 2 10 14 "AKAZE" &
    run_split 3 15 19 "AKAZE" &

elif [ $N_SPLITS -eq 2 ]; then
    # For 2 cores: SIFT vs AKAZE
    echo "Running in 2 parallel processes..."
    echo "  - SIFT 0-9 (10 configs)"
    echo "  - AKAZE 10-19 (10 configs)"
    echo ""

    run_split 0 0 9 "SIFT" &
    run_split 1 10 19 "AKAZE" &

elif [ $N_SPLITS -ge 10 ]; then
    # For many cores: 1 SIFT + 1 AKAZE per 2 cores
    echo "Running in $N_SPLITS parallel processes..."
    echo "  - Each SIFT config on separate core (0-9)"
    echo "  - Each AKAZE config on separate core (10-19)"
    echo ""

    # Run each config separately
    for i in {0..9}; do
        run_split $i $i $i "SIFT" &
    done
    for i in {10..19}; do
        run_split $i $i $i "AKAZE" &
    done

else
    # Generic split for other values
    echo "Running in $N_SPLITS parallel processes..."
    CONFIGS_PER_SPLIT=$(( (20 + N_SPLITS - 1) / N_SPLITS ))

    for i in $(seq 0 $((N_SPLITS - 1))); do
        start_idx=$((i * CONFIGS_PER_SPLIT))
        end_idx=$((start_idx + CONFIGS_PER_SPLIT - 1))

        # Don't exceed total configs
        if [ $end_idx -ge 20 ]; then
            end_idx=19
        fi

        # Skip if start_idx is beyond total configs
        if [ $start_idx -ge 20 ]; then
            break
        fi

        # Determine descriptor type
        if [ $start_idx -lt 10 ]; then
            descriptor="SIFT"
        else
            descriptor="AKAZE"
        fi

        run_split "$i" "$start_idx" "$end_idx" "$descriptor" &
    done
fi

# Wait for all processes to complete
echo ""
echo "============================================"
echo "Waiting for all processes to complete..."
echo "You can monitor progress with:"
echo "  tail -f $LOGS_DIR/*.log"
echo "============================================"
echo ""

wait

echo ""
echo "============================================"
echo "All parallel processes completed!"
echo "============================================"
echo ""
echo "Check individual logs in: $LOGS_DIR/"
echo "View results in W&B project: OPTIONAL1"
echo ""
echo "To analyze results:"
echo "  grep -r 'Test set accuracy' $LOGS_DIR/"
echo "  grep -r 'BEST CONFIGURATION' $LOGS_DIR/"
