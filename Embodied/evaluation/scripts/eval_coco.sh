#!/usr/bin/env bash
# LocateAnything - COCO Detection Evaluation Pipeline
# Steps: DDP Inference → Format Conversion → COCO AP Evaluation → Speed Analysis
set -x

# ==================== DDP Configuration ====================
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}
TOTAL_GPUS=$((GPUS * NNODES))

# ==================== Model Configuration ====================
MODEL_PATH=${MODEL_PATH:-"path/to/Embodied"}
export HF_TOKEN="${HF_TOKEN:-}"

# ==================== Dataset Configuration ====================
TEST_JSONL_PATH=${TEST_JSONL_PATH:-"path/to/EvalData/_annotations/box_eval/COCO.jsonl"}
IMAGE_ROOT_DIR=${IMAGE_ROOT_DIR:-"path/to/EvalData/"}
COCO_JSON=${COCO_JSON:-"path/to/EvalData/coco/instances_val2017.json"}

# ==================== Inference Parameters ====================
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
NUM_WORKERS=${NUM_WORKERS:-4}
GENERATION_MODE=${GENERATION_MODE:-"hybrid"} # Options: fast, slow, hybrid
OUTPUT_DIR_OVERRIDE=""

# ==================== Help ====================
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  --model_path PATH     Path to model"
    echo "  --generation_mode M   fast | slow | hybrid (default: hybrid)"
    echo "  --image_root DIR      Image root directory"
    echo "  --test_jsonl PATH     Override COCO test JSONL path"
    echo "  --coco_json PATH      Override COCO annotation JSON"
    echo "  --output_dir DIR      Override output directory"
    echo ""
    echo "Example:"
    echo "  GPUS=8 bash $0"
    echo "  bash $0 --model_path /path/to/model --generation_mode fast"
}

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)       MODEL_PATH="$2"; shift 2;;
        --generation_mode)  GENERATION_MODE="$2"; shift 2;;
        --output_dir)       OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
        --image_root)       IMAGE_ROOT_DIR="$2"; shift 2;;
        --test_jsonl)       TEST_JSONL_PATH="$2"; shift 2;;
        --coco_json)        COCO_JSON="$2"; shift 2;;
        -h|--help)          print_help; exit 0;;
        *)                  echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

# ==================== Paths ====================
MODEL_NAME=$(basename "${MODEL_PATH%/}")
EAGLE_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE/${GENERATION_MODE}"
else
    OUTPUT_DIR="${EAGLE_BASE}/results/${MODEL_NAME}/coco/${GENERATION_MODE}"
fi
SAVE_PATH="$OUTPUT_DIR/eval_results.jsonl"
FASTEVAL_TSV="$OUTPUT_DIR/fast_eval.tsv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/evaluation_log_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

# ==================== NCCL ====================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ==================== Print Configuration ====================
script_name=$(basename "${BASH_SOURCE[0]}")

echo "=========================================="
echo "=== LocateAnything COCO Evaluation ==="
echo "=========================================="
echo "NNODES: $NNODES | GPUS: $GPUS | TOTAL: $TOTAL_GPUS"
echo "NODE_RANK: $NODE_RANK | MASTER: $MASTER_ADDR:$PORT"
echo "MODEL_PATH: $MODEL_PATH"
echo "GENERATION_MODE: $GENERATION_MODE"
echo "TEST_JSONL: $TEST_JSONL_PATH"
echo "COCO_JSON: $COCO_JSON"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SCRIPT: $script_name"
echo "=========================================="

# ==================== GPU Check ====================
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found"; exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $AVAILABLE_GPUS -lt $GPUS ]; then
    echo "Warning: Only $AVAILABLE_GPUS GPUs available, less than requested $GPUS. Using all available."
    GPUS=$AVAILABLE_GPUS
    TOTAL_GPUS=$((GPUS * NNODES))
fi

# ==================== Paths ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

# ==================== Step 1: DDP Inference ====================
echo ""
echo "Step 1: Running DDP Inference..."
echo "=========================================="

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$EVAL_DIR/inference_detection_ddp.py" \
    --world_size $TOTAL_GPUS \
    --num_nodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $PORT \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL_PATH" \
    --image_root_dir "$IMAGE_ROOT_DIR" \
    --save_path "$SAVE_PATH" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_workers $NUM_WORKERS \
    --generation_mode "$GENERATION_MODE" \
    2>&1 | tee -a "$LOG_FILE"

INFERENCE_EXIT_CODE=${PIPESTATUS[0]}
if [ $INFERENCE_EXIT_CODE -ne 0 ]; then
    echo "Inference failed with exit code: $INFERENCE_EXIT_CODE"
    exit $INFERENCE_EXIT_CODE
fi

echo "Inference completed. Predictions: $SAVE_PATH"

# ==================== Step 2: Format Conversion ====================
echo ""
echo "Step 2: Converting predictions to FastEval TSV format..."
echo "=========================================="

python "$EVAL_DIR/utils/convert_coco_lvis_to_standard_format.py" \
    --our_pred_jsonl "$SAVE_PATH" \
    --coco_json "$COCO_JSON" \
    --out_tsv "$FASTEVAL_TSV" \
    --positive_only

CONVERT_EXIT_CODE=$?
if [ $CONVERT_EXIT_CODE -ne 0 ]; then
    echo "Format conversion failed with exit code: $CONVERT_EXIT_CODE"
    exit $CONVERT_EXIT_CODE
fi

echo "Format conversion completed."

# ==================== Step 3: COCO AP Evaluation ====================
echo ""
echo "Step 3: Running COCO AP evaluation..."
echo "=========================================="

python "$EVAL_DIR/metrics/coco_lvis_metric.py" \
    --gt "$COCO_JSON" \
    --pred_tsv "$FASTEVAL_TSV"

EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo "Evaluation completed."

# ==================== Step 4: Speed Analysis (BPS/TPS) ====================
echo ""
echo "Step 4: Analyzing speed (TPS, BPS)..."
echo "=========================================="

python "$EVAL_DIR/metrics/analyze_speed.py" \
    --log_file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "COCO Evaluation Pipeline completed!"
echo "=========================================="
echo "  Predictions: $SAVE_PATH"
echo "  FastEval TSV: $FASTEVAL_TSV"
echo "  Log: $LOG_FILE"
echo "=========================================="
