#!/bin/bash
# ODesignBench copy of run_af3_a800.sh with /root/af_output bind mount fix.
# AF3 resources are from the original path.
# This script runs AlphaFold3 using the AF3 Docker image (not apptainer/sif).

# Parse arguments from refold_api.py
# $1: exp_name
# $2: input_json
# $3: output_dir
# $4: gpus (comma-separated, e.g., "0,1,2,3")
# $5: run_data_pipeline (True/False)
# $6: cache_dir
# $7: num_diffusion_samples (optional, default: 1)

exp_name="$1"
input_json="$2"
output_path="$3"
gpus="$4"
run_data_pipeline="$5"
cache_dir="$6"
num_diffusion_samples="${7:-1}"  # Default to 1 if not provided

# AF3 installation directories (override via env vars)
AF3_BASE="${AF3_BASE:-/path/to/af3}"
AF3_PUBLIC_DB="${AF3_PUBLIC_DB:-/path/to/public_databases}"

# AF3 Docker image name (override via env var if your image tag differs)
AF3_DOCKER_IMAGE="${AF3_DOCKER_IMAGE:-alphafold3}"

# Create output directory and log directory (for debugging)
mkdir -p "$output_path"
AF3_LOG_DIR="$(dirname "$output_path")/af3_log"
mkdir -p "$AF3_LOG_DIR"

# Convert comma-separated gpus to space-separated for launch.py
gpus_space_separated=${gpus//,/ }

# Build docker volume mounts
# - /root/af_output: some containers use this default output path
# - /app/alphafold/log: capture AF3 logs for debugging (see af3_log/ after run)
BIND_MOUNTS=(
    -v "$AF3_BASE/model:/root/models"
    -v "$AF3_PUBLIC_DB:/root/public_databases"
    -v "$output_path:$output_path"
    -v "$output_path:/root/af_output"
    -v "$AF3_LOG_DIR:/app/alphafold/log"
    -v "$input_json:$input_json"
)

if [[ "$AF3_BASE" == "/path/to/af3" || ! -d "$AF3_BASE/model" ]]; then
    echo "ERROR: Set AF3_BASE to your AlphaFold3 installation directory (must contain model/ for docker volume mounts)." >&2
    exit 1
fi

if [[ "$AF3_PUBLIC_DB" == "/path/to/public_databases" || ! -d "$AF3_PUBLIC_DB" ]]; then
    echo "ERROR: Set AF3_PUBLIC_DB to your AlphaFold3 public_databases directory." >&2
    exit 1
fi

# Optional bind for local home paths referenced by input/output files.
if [[ -n "${HOME:-}" && -d "${HOME:-}" ]]; then
    BIND_MOUNTS+=(-v "${HOME}:${HOME}")
fi

# Add cache_dir bind mount if provided
if [[ -n "$cache_dir" && "$cache_dir" != "None" && "$cache_dir" != "" ]]; then
    mkdir -p "$cache_dir"
    BIND_MOUNTS+=(-v "$cache_dir:$cache_dir")
fi

# Limit JAX GPU memory preallocation to reduce OOM risk.
# Use the non-deprecated env var to avoid conflicts inside AF3/JAX runtime.
# If users still export XLA_PYTHON_CLIENT_MEM_FRACTION, treat it as fallback.
if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION"
fi
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.6}"
unset XLA_PYTHON_CLIENT_MEM_FRACTION

# Execute AlphaFold3 using Docker
docker run --rm \
    --gpus all \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION" \
    -e PROMPT_COMMAND= \
    "${BIND_MOUNTS[@]}" \
    "$AF3_DOCKER_IMAGE" \
    python /app/alphafold/launch.py \
    --input_json "$input_json" \
    --output_dir "$output_path" \
    --run_data_pipeline "$run_data_pipeline" \
    --gpus $gpus_space_separated \
    --exp_name "$exp_name" \
    --num_diffusion_samples "$num_diffusion_samples"
