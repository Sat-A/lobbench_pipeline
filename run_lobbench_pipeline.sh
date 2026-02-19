#!/bin/bash
set -euo pipefail

# LOBbench Automation Pipeline
# Submits inference (GPU) + scoring (CPU) SLURM jobs for a given checkpoint.
#
# Default (HF mode):  Generate exactly the same samples as the HF baseline,
#                     then score + compare.
# Custom mode:        Generate N random samples, score only.
#
# Usage:
#   ./pipeline/run_lobbench_pipeline.sh <CKPT_PATH> [OPTIONS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================
# Defaults
# ============================================================
NAME=""
CHECKPOINT_STEP=""
STOCKS="GOOG INTC"
BATCH_SIZE=64
N_COND_MSGS=500
N_GEN_MSGS=500
NO_HF_COMPARE=0
N_SEQUENCES=1024
INFER_WALLTIME="06:00:00"
BENCH_WALLTIME="24:00:00"
SKIP_INFERENCE=0
INFERENCE_DIR=""

# ============================================================
# Parse arguments
# ============================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <CKPT_PATH> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  CKPT_PATH                  Checkpoint directory path"
    echo ""
    echo "Options:"
    echo "  --name NAME                Run name (default: derived from checkpoint dirname)"
    echo "  --checkpoint_step N        Step to load (default: auto-detect latest)"
    echo "  --stocks \"GOOG INTC\"       Stocks to evaluate (default: \"GOOG INTC\")"
    echo "  --batch_size N             Inference batch size (default: 64)"
    echo "  --n_cond_msgs N            Conditioning messages (default: 500)"
    echo "  --n_gen_msgs N             Generated messages (default: 500)"
    echo "  --no_hf_compare            Custom mode: random sampling, no HF comparison"
    echo "  --n_sequences N            Custom mode only: number of sequences (default: 1024)"
    echo "  --infer_walltime T         Inference walltime (default: 06:00:00)"
    echo "  --bench_walltime T         Benchmarking walltime (default: 24:00:00)"
    echo "  --skip_inference           Reuse existing inference (needs --inference_dir)"
    echo "  --inference_dir DIR        Path to existing inference results"
    exit 1
fi

CKPT_PATH="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --name)             NAME="$2";             shift 2 ;;
        --checkpoint_step)  CHECKPOINT_STEP="$2";  shift 2 ;;
        --stocks)           STOCKS="$2";           shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";       shift 2 ;;
        --n_cond_msgs)      N_COND_MSGS="$2";      shift 2 ;;
        --n_gen_msgs)       N_GEN_MSGS="$2";       shift 2 ;;
        --no_hf_compare)    NO_HF_COMPARE=1;       shift 1 ;;
        --n_sequences)      N_SEQUENCES="$2";      shift 2 ;;
        --infer_walltime)   INFER_WALLTIME="$2";   shift 2 ;;
        --bench_walltime)   BENCH_WALLTIME="$2";   shift 2 ;;
        --skip_inference)   SKIP_INFERENCE=1;       shift 1 ;;
        --inference_dir)    INFERENCE_DIR="$2";    shift 2 ;;
        *) echo "ERROR: Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# Load config
# ============================================================
CONFIG="${SCRIPT_DIR}/config.sh"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: ${CONFIG} not found."
    echo "  cp pipeline/config.sh.template pipeline/config.sh"
    echo "  # then edit paths in config.sh"
    exit 1
fi
source "$CONFIG"

# ============================================================
# Validate checkpoint
# ============================================================
if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

# Auto-detect checkpoint step (latest numeric subdirectory)
if [ -z "$CHECKPOINT_STEP" ]; then
    CHECKPOINT_STEP=$(ls -1d "${CKPT_PATH}"/[0-9]* 2>/dev/null | xargs -I{} basename {} | sort -n | tail -1 || echo "")
    if [ -z "$CHECKPOINT_STEP" ]; then
        echo "ERROR: No checkpoint steps found in ${CKPT_PATH}"
        echo "  Use --checkpoint_step N to specify manually"
        exit 1
    fi
fi
echo "[*] Checkpoint step: ${CHECKPOINT_STEP}"

# Derive run name from checkpoint dirname (strip wandb hash suffix)
if [ -z "$NAME" ]; then
    CKPT_BASENAME=$(basename "$CKPT_PATH")
    # Strip trailing _<wandb_hash> (8 chars after underscore)
    NAME=$(echo "$CKPT_BASENAME" | sed 's/_[a-z0-9]\{6,10\}$//')
fi
echo "[*] Run name: ${NAME}"

# ============================================================
# Validate data directories per stock
# ============================================================
VALID_STOCKS=""
for STOCK in $STOCKS; do
    DATA_VAR="${STOCK}_DATA"
    DATA_DIR="${!DATA_VAR:-}"
    if [ -z "$DATA_DIR" ]; then
        echo "WARNING: No data directory configured for ${STOCK} (set ${DATA_VAR} in config.sh)"
        continue
    fi
    if [ ! -d "$DATA_DIR" ]; then
        echo "WARNING: Data directory not found for ${STOCK}: ${DATA_DIR}"
        continue
    fi
    VALID_STOCKS="${VALID_STOCKS} ${STOCK}"
done
VALID_STOCKS=$(echo "$VALID_STOCKS" | xargs)  # trim whitespace

if [ -z "$VALID_STOCKS" ]; then
    echo "ERROR: No valid stocks to evaluate"
    exit 1
fi
echo "[*] Stocks: ${VALID_STOCKS}"

# ============================================================
# HF index extraction (default mode)
# ============================================================
extract_hf_indices() {
    local stock="$1"
    local hf_gen="${REPO_DIR}/lob_bench/hf_data_git/${stock}/data_gen_lobs5"
    local out_file="${SCRIPT_DIR}/.hf_indices_${stock}.txt"

    if [ ! -d "$hf_gen" ]; then
        echo "WARNING: No HF data for ${stock} at ${hf_gen}"
        return 1
    fi

    ls "${hf_gen}"/*message*.csv 2>/dev/null \
        | xargs -n1 basename \
        | sed 's/.*real_id_//' | sed 's/_gen.*//' \
        | sort -nu > "$out_file"

    local count=$(wc -l < "$out_file")
    echo "[*] HF indices for ${stock}: ${count} samples"
    return 0
}

# ============================================================
# Skip inference validation
# ============================================================
if [ "$SKIP_INFERENCE" -eq 1 ]; then
    if [ -z "$INFERENCE_DIR" ]; then
        echo "ERROR: --skip_inference requires --inference_dir"
        exit 1
    fi
    if [ ! -d "$INFERENCE_DIR" ]; then
        echo "ERROR: Inference dir not found: ${INFERENCE_DIR}"
        exit 1
    fi
fi

# ============================================================
# Create logs directory
# ============================================================
LOGS_DIR="${REPO_DIR}/logs"
mkdir -p "$LOGS_DIR"

# ============================================================
# Submit jobs per stock
# ============================================================
echo ""
echo "=============================================="
echo "LOBbench Pipeline: ${NAME}"
echo "=============================================="
echo "Checkpoint: ${CKPT_PATH} (step ${CHECKPOINT_STEP})"
echo "Mode: $([ "$NO_HF_COMPARE" -eq 1 ] && echo "Custom (${N_SEQUENCES} random)" || echo "HF (matched samples)")"
echo "Config: ${N_COND_MSGS} cond + ${N_GEN_MSGS} gen, batch ${BATCH_SIZE}"
echo "=============================================="
echo ""

ALL_BENCH_JOBS=""

for STOCK in $VALID_STOCKS; do
    DATA_VAR="${STOCK}_DATA"
    DATA_DIR="${!DATA_VAR}"

    echo "--- ${STOCK} ---"

    # HF mode: extract indices
    SAMPLE_INDICES_FILE=""
    SKIP_HF_COMPARE="$NO_HF_COMPARE"
    if [ "$NO_HF_COMPARE" -eq 0 ]; then
        if extract_hf_indices "$STOCK"; then
            SAMPLE_INDICES_FILE="${SCRIPT_DIR}/.hf_indices_${STOCK}.txt"
        else
            echo "  Falling back to custom mode for ${STOCK} (no HF data)"
            SKIP_HF_COMPARE=1
        fi
    fi

    # --------------------------------------------------------
    # Inference job
    # --------------------------------------------------------
    if [ "$SKIP_INFERENCE" -eq 0 ]; then
        INFER_JOB_ID=$(sbatch --parsable \
            --job-name="infer_${NAME}_${STOCK}" \
            --time="${INFER_WALLTIME}" \
            --output="${LOGS_DIR}/infer_${NAME}_${STOCK}_%j.out" \
            --error="${LOGS_DIR}/infer_${NAME}_${STOCK}_%j.err" \
            --partition="${PARTITION}" \
            --export=ALL,REPO_DIR="${REPO_DIR}",PYTHON="${PYTHON}",STOCK="${STOCK}",DATA_DIR="${DATA_DIR}",CKPT_PATH="${CKPT_PATH}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",RUN_NAME="${NAME}",BATCH_SIZE="${BATCH_SIZE}",N_COND_MSGS="${N_COND_MSGS}",N_GEN_MSGS="${N_GEN_MSGS}",N_SEQUENCES="${N_SEQUENCES}",SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE}",SKIP_HF_COMPARE="${SKIP_HF_COMPARE}",NTFY_TOPIC_INFERENCE="${NTFY_TOPIC_INFERENCE}" \
            "${SCRIPT_DIR}/_infer.batch")

        echo "  Inference job: ${INFER_JOB_ID}"

        # Infer output dir (matches _infer.batch convention)
        INFER_OUTPUT="${REPO_DIR}/LOBS5/inference_results/${NAME}_${STOCK}_${INFER_JOB_ID}"
        DEPENDENCY="--dependency=afterok:${INFER_JOB_ID}"
    else
        echo "  Skipping inference (reusing ${INFERENCE_DIR})"
        INFER_OUTPUT="${INFERENCE_DIR}"
        INFER_JOB_ID="skipped"
        DEPENDENCY=""
    fi

    # --------------------------------------------------------
    # Bench job (depends on inference)
    # --------------------------------------------------------
    BENCH_JOB_ID=$(sbatch --parsable \
        ${DEPENDENCY} \
        --job-name="bench_${NAME}_${STOCK}" \
        --time="${BENCH_WALLTIME}" \
        --output="${LOGS_DIR}/bench_${NAME}_${STOCK}_%j.out" \
        --error="${LOGS_DIR}/bench_${NAME}_${STOCK}_%j.err" \
        --partition="${PARTITION}" \
        --export=ALL,REPO_DIR="${REPO_DIR}",PYTHON="${PYTHON}",STOCK="${STOCK}",RUN_NAME="${NAME}",INFER_OUTPUT="${INFER_OUTPUT}",SKIP_HF_COMPARE="${SKIP_HF_COMPARE}",NTFY_TOPIC_BENCHMARKS="${NTFY_TOPIC_BENCHMARKS}" \
        "${SCRIPT_DIR}/_bench.batch")

    echo "  Bench job:     ${BENCH_JOB_ID} (depends on ${INFER_JOB_ID})"
    ALL_BENCH_JOBS="${ALL_BENCH_JOBS} ${BENCH_JOB_ID}"
    echo ""
done

# ============================================================
# Summary
# ============================================================
echo "=============================================="
echo "Pipeline submitted successfully!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue --me"
echo ""
echo "Results will be at:"
echo "  ${REPO_DIR}/results_${NAME}/"
echo ""
echo "Bench job IDs:${ALL_BENCH_JOBS}"
