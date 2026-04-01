#!/bin/bash
# Run AlphaFold3 via Singularity/Apptainer for HPC clusters where Docker is unavailable.
# Arguments are compatible with refold/run_af3.sh:
#   $1 exp_name
#   $2 input_json
#   $3 output_dir
#   $4 gpus (comma-separated, e.g. "0,1")
#   $5 run_data_pipeline (True/False)
#   $6 cache_dir
#   $7 num_diffusion_samples (optional, default: 1)

set -euo pipefail

exp_name="$1"
input_json="$2"
output_path="$3"
gpus="$4"
run_data_pipeline="$5"
cache_dir="$6"
num_diffusion_samples="${7:-1}"

AF3_BASE="${AF3_BASE:-/path/to/af3}"
AF3_PUBLIC_DB="${AF3_PUBLIC_DB:-/path/to/public_databases}"
AF3_SIF_IMAGE="${AF3_SIF_IMAGE:-/path/to/alphafold3.sif}"
AF3_DIALECT_PATCH="${AF3_DIALECT_PATCH:-true}"
AF3_ASSETS="${AF3_ASSETS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../assets}"

if command -v singularity >/dev/null 2>&1; then
    AF3_CONTAINER_CMD="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    AF3_CONTAINER_CMD="apptainer"
else
    echo "ERROR: Neither 'singularity' nor 'apptainer' is available in PATH." >&2
    exit 1
fi

if [[ "$AF3_BASE" == "/path/to/af3" || ! -d "$AF3_BASE/models" ]]; then
    echo "ERROR: Set AF3_BASE to your AlphaFold3 directory (must contain models/)." >&2
    exit 1
fi

if [[ "$AF3_PUBLIC_DB" == "/path/to/public_databases" || ! -d "$AF3_PUBLIC_DB" ]]; then
    echo "ERROR: Set AF3_PUBLIC_DB to your AlphaFold3 public_databases directory." >&2
    exit 1
fi

if [[ "$AF3_SIF_IMAGE" == "/path/to/alphafold3.sif" || ! -f "$AF3_SIF_IMAGE" ]]; then
    echo "ERROR: Set AF3_SIF_IMAGE to your AlphaFold3 .sif image path." >&2
    exit 1
fi

mkdir -p "$output_path"
AF3_LOG_DIR="$(dirname "$output_path")/af3_log"
mkdir -p "$AF3_LOG_DIR"

BIND_MOUNTS=(
    -B "$AF3_BASE/models:/root/models"
    -B "$AF3_PUBLIC_DB:/root/public_databases"
    -B "$output_path:$output_path"
    -B "$output_path:/root/af_output"
    -B "$AF3_LOG_DIR:/app/alphafold/log"
    -B "$input_json:$input_json"
    -B "$AF3_ASSETS:/assets"
)

if [[ -n "${HOME:-}" && -d "${HOME:-}" ]]; then
    BIND_MOUNTS+=(-B "${HOME}:${HOME}")
fi

if [[ -n "$cache_dir" && "$cache_dir" != "None" && "$cache_dir" != "" ]]; then
    mkdir -p "$cache_dir"
    BIND_MOUNTS+=(-B "$cache_dir:$cache_dir")
fi

if [[ -z "${XLA_CLIENT_MEM_FRACTION:-}" && -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    export XLA_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION"
fi
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.6}"
unset XLA_PYTHON_CLIENT_MEM_FRACTION
export SINGULARITYENV_XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
export APPTAINERENV_XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"

if [[ -n "$gpus" && "$gpus" != "all" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpus"
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpus"
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpus"
fi

# Optional patch for alphafoldserver dialect to allow MSA path injection.
PATCHED_FILE="/tmp/alphafold3_folding_input_patched.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$AF3_DIALECT_PATCH" == "true" ]]; then
    if [[ ! -f "$PATCHED_FILE" ]]; then
        echo "Generating AF3 dialect patch (one-time, cached at $PATCHED_FILE)..."
        "$AF3_CONTAINER_CMD" exec --nv \
            -B /tmp:/tmp \
            -B "$SCRIPT_DIR:/opt/odesign_patch" \
            "$AF3_SIF_IMAGE" \
            python3 /opt/odesign_patch/gen_af3_patch.py
        echo "Patch generated successfully."
    fi
fi

if [[ "$AF3_DIALECT_PATCH" == "true" && -f "$PATCHED_FILE" ]]; then
    BIND_MOUNTS+=(-B "$PATCHED_FILE:/app/alphafold/src/alphafold3/common/folding_input.py")
fi

"$AF3_CONTAINER_CMD" exec --nv \
    "${BIND_MOUNTS[@]}" \
    "$AF3_SIF_IMAGE" \
    python /app/alphafold/run_alphafold.py \
    --json_path "$input_json" \
    --output_dir "$output_path" \
    --run_data_pipeline="$run_data_pipeline" \
    --gpu_device 0 \
    --num_diffusion_samples "$num_diffusion_samples"

