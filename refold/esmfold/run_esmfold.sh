#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Script to run distributed ESMFold processing using torchrun

# export CUDA_VISIBLE_DEVICES=6,7

get_free_port() {
    ${PYTHON:-python} -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

is_port_available() {
    local port="$1"
    ${PYTHON:-python} - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

FREE_PORT=$(get_free_port)

# Default values
NUM_GPUS=2
NAME=""
SEQUENCES_FILE=""
OUTPUT_DIR=""
VERBOSE_GPU=false
MASTER_PORT=$FREE_PORT
ESMFOLD_MODEL_DIR=""

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    # echo "  -n, --name NAME              Name for the output files (required)"
    echo "  -s, --sequences FILE         JSON file containing sequences (required)"
    echo "  -o, --output_dir DIR         Output directory for PDB files (required)"
    echo "  -g, --num_gpus NUM           Number of GPUs to use (default: 2)"
    echo "  -p, --master_port PORT       Master port for distributed training (default: auto free port)"
    echo "  -m, --model_dir DIR          ESMFold model directory (default: ./weights)"
    echo "  -v, --verbose_gpu            Enable verbose GPU memory reporting"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n my_protein -s sequences.json -o ./output -g 4 -v"
    echo "  $0 --name test --sequences seqs.json --output_dir results --num_gpus 8"
    echo ""
    echo "Prerequisites:"
    echo "  - PyTorch with distributed support"
    echo "  - ESMFold model in ./ESMFold directory"
    echo "  - Sequences file in JSON format (list of strings)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # -n|--name)
        #     NAME="$2"
        #     shift 2
        #     ;;
        -s|--sequences)
            SEQUENCES_FILE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -p|--master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -m|--model_dir)
            ESMFOLD_MODEL_DIR="$2"
            shift 2
            ;;
        -v|--verbose_gpu)
            VERBOSE_GPU=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
# if [[ -z "$NAME" ]]; then
#     echo "Error: --name is required"
#     show_usage
#     exit 1
# fi

if [[ -z "$SEQUENCES_FILE" ]]; then
    echo "Error: --sequences is required"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    show_usage
    exit 1
fi

# Validate files and directories
if [[ ! -f "$SEQUENCES_FILE" ]]; then
    echo "Error: Sequences file '$SEQUENCES_FILE' not found"
    exit 1
fi

# Set default model directory if not provided
if [[ -z "$ESMFOLD_MODEL_DIR" ]]; then
    ESMFOLD_MODEL_DIR="$(dirname "$0")/weights"
fi

# Validate model directory
if [[ ! -d "$ESMFOLD_MODEL_DIR" ]]; then
    echo "Error: ESMFold model directory '$ESMFOLD_MODEL_DIR' not found. Please ensure ESMFold model is available."
    exit 1
fi

# Validate number of GPUs
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Error: Number of GPUs must be a positive integer"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Make sure CUDA is available."
fi

# Check available GPUs
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    if [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    fi
fi

# If the requested port is occupied, automatically switch to a free one.
if ! is_port_available "$MASTER_PORT"; then
    FALLBACK_PORT=$(get_free_port)
    echo "Warning: master_port $MASTER_PORT is in use, fallback to free port $FALLBACK_PORT"
    MASTER_PORT="$FALLBACK_PORT"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build torchrun command (use python -m to avoid bad shebang in torchrun script)
TORCHRUN_CMD="${PYTHON:-python} -m torch.distributed.run"
TORCHRUN_CMD+=" --nproc_per_node=$NUM_GPUS"
TORCHRUN_CMD+=" --nnodes=1"
TORCHRUN_CMD+=" --node_rank=0"
TORCHRUN_CMD+=" --master_addr=localhost"
TORCHRUN_CMD+=" --master_port=$MASTER_PORT"
TORCHRUN_CMD+=" $(dirname "$0")/run_esmfold_distributed.py"
# TORCHRUN_CMD+=" --name '$NAME'"
TORCHRUN_CMD+=" --sequences '$SEQUENCES_FILE'"
TORCHRUN_CMD+=" --output_dir '$OUTPUT_DIR'"
TORCHRUN_CMD+=" --esmfold_model_dir '$ESMFOLD_MODEL_DIR'"

if [[ "$VERBOSE_GPU" == true ]]; then
    TORCHRUN_CMD+=" --verbose_gpu"
fi

# Print configuration
echo "=== Distributed ESMFold Configuration ==="
echo "Name: $NAME"
echo "Sequences file: $SEQUENCES_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo "ESMFold model directory: $ESMFOLD_MODEL_DIR"
echo "Verbose GPU: $VERBOSE_GPU"
echo "========================================="
echo ""

# Count sequences
if command -v python3 &> /dev/null; then
    SEQ_COUNT=$(python3 -c "import json; print(len(json.load(open('$SEQUENCES_FILE'))))" 2>/dev/null || echo "unknown")
    echo "Number of sequences: $SEQ_COUNT"
    echo ""
fi

# Run the command
echo "Running distributed ESMFold..."
echo "Command: $TORCHRUN_CMD"
echo ""

# Execute the command
eval $TORCHRUN_CMD
EXIT_CODE=$?

# Check results
if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "=== Distributed ESMFold completed successfully! ==="
    
    # Check for results file
    RESULTS_FILE="$OUTPUT_DIR/${NAME}_results.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        echo "Results file: $RESULTS_FILE"
        if command -v python3 &> /dev/null; then
            RESULT_COUNT=$(python3 -c "import json; print(len(json.load(open('$RESULTS_FILE'))))" 2>/dev/null || echo "unknown")
            echo "Generated PDB files: $RESULT_COUNT"
        fi
    fi
    
    echo "Output directory: $OUTPUT_DIR"
    echo "==============================================="
else
    echo ""
    echo "=== Distributed ESMFold failed with exit code $EXIT_CODE ==="
    echo "Please check the error messages above for troubleshooting."
fi

exit $EXIT_CODE 