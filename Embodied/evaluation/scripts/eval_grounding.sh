
#!/usr/bin/env bash
# LocateAnything - Grounding Benchmark Evaluation Pipeline
# Supports: HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc,
#           RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone, FSCD_test
# Steps: DDP Inference → Metrics Evaluation → Speed Analysis
set -x

# ==================== DDP Configuration ====================
GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}
TOTAL_GPUS=$((GPUS * NNODES))

# ==================== Model Configuration ====================
MODEL_PATH=${MODEL_PATH:-"/path/to/LocateAnything"}
export HF_TOKEN="${HF_TOKEN:-}"

# ==================== Dataset Configuration ====================
DATASET=${DATASET:-"SROIE"} # Options: HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc, RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone, FSCD_test
EVAL_TYPE=${EVAL_TYPE:-"box_eval"} # Options: box_eval, point_eval,
IMAGE_ROOT_DIR=${IMAGE_ROOT_DIR:-"path/to/EvalData/"}

# ==================== Inference Parameters ====================
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM_WORKERS=${NUM_WORKERS:-4}
GENERATION_MODE=${GENERATION_MODE:-"hybrid"} # Options: fast, slow, hybrid

# Optional overrides
TEST_JSONL_PATH=""
OUTPUT_DIR_OVERRIDE=""

# ==================== Help ====================
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  --dataset NAME        Dataset (default: SROIE)"
    echo "                        Options: HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc,"
    echo "                                 RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone, FSCD_test"
    echo "  --eval_type TYPE      Evaluation type: box_eval, point_eval"
    echo "  --model_path PATH     Path to model"
    echo "  --generation_mode M   fast | slow | hybrid (default: hybrid)"
    echo "  --image_root DIR      Image root directory"
    echo "  --test_jsonl PATH     Override test JSONL path"
    echo "  --output_dir DIR      Override output directory"
    echo ""
    echo "Example:"
    echo "  DATASET=HierText GPUS=8 bash $0"
    echo "  bash $0 --dataset RefCOCOg_val --generation_mode fast"
}

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)          DATASET="$2"; shift 2;;
        --model_path)       MODEL_PATH="$2"; shift 2;;
        --image_root)       IMAGE_ROOT_DIR="$2"; shift 2;;
        --test_jsonl)       TEST_JSONL_PATH="$2"; shift 2;;
        --output_dir)       OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
        --output_base)      OUTPUT_BASE_OVERRIDE="$2"; shift 2;;
        --eval_type)        EVAL_TYPE="$2"; shift 2;;
        --generation_mode)  GENERATION_MODE="$2"; shift 2;;
        -h|--help)          print_help; exit 0;;
        *)                  echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

# ==================== Paths ====================
MODEL_NAME=$(basename "${MODEL_PATH%/}")
EAGLE_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANNO_BASE="${IMAGE_ROOT_DIR}/_annotations/${EVAL_TYPE}"


if [[ -n "$OUTPUT_BASE_OVERRIDE" ]]; then
    OUTPUT_BASE="$OUTPUT_BASE_OVERRIDE"
else
    OUTPUT_BASE="${EAGLE_BASE}/results/${MODEL_NAME}/${EVAL_TYPE}"
fi


# ==================== Dataset-specific Configuration ====================
case "$DATASET" in
    LVIS)           DEFAULT_JSONL="$ANNO_BASE/LVIS.jsonl";          OUTPUT_DIR="$OUTPUT_BASE/LVIS";;
    COCO)           DEFAULT_JSONL="$ANNO_BASE/COCO.jsonl";          OUTPUT_DIR="$OUTPUT_BASE/COCO";;
    HierText)       DEFAULT_JSONL="$ANNO_BASE/HierText.jsonl";       OUTPUT_DIR="$OUTPUT_BASE/HierText";;
    DocLayNet)      DEFAULT_JSONL="$ANNO_BASE/DocLayNet.jsonl";      OUTPUT_DIR="$OUTPUT_BASE/DocLayNet";;
    HumanRef)       DEFAULT_JSONL="$ANNO_BASE/HumanRef.jsonl";      OUTPUT_DIR="$OUTPUT_BASE/HumanRef";;
    Dense200)       DEFAULT_JSONL="$ANNO_BASE/Dense200.jsonl";      OUTPUT_DIR="$OUTPUT_BASE/Dense200";;
    IC15)           DEFAULT_JSONL="$ANNO_BASE/IC15.jsonl";          OUTPUT_DIR="$OUTPUT_BASE/IC15";;
    M6Doc)          DEFAULT_JSONL="$ANNO_BASE/M6Doc.jsonl";         OUTPUT_DIR="$OUTPUT_BASE/M6Doc";;
    RefCOCOg_test)  DEFAULT_JSONL="$ANNO_BASE/RefCOCOg_test.jsonl"; OUTPUT_DIR="$OUTPUT_BASE/RefCOCOg_test";;
    RefCOCOg_val)   DEFAULT_JSONL="$ANNO_BASE/RefCOCOg_val.jsonl";  OUTPUT_DIR="$OUTPUT_BASE/RefCOCOg_val";;
    SROIE)          DEFAULT_JSONL="$ANNO_BASE/SROIE.jsonl";         OUTPUT_DIR="$OUTPUT_BASE/SROIE";;
    TotalText)      DEFAULT_JSONL="$ANNO_BASE/TotalText.jsonl";     OUTPUT_DIR="$OUTPUT_BASE/TotalText";;
    VisDrone)        DEFAULT_JSONL="$ANNO_BASE/VisDrone.jsonl";      OUTPUT_DIR="$OUTPUT_BASE/VisDrone";;
    FSCD_test)      DEFAULT_JSONL="$ANNO_BASE/FSCD_test.jsonl";     OUTPUT_DIR="$OUTPUT_BASE/FSCD_test";;
    *)
        echo "Unsupported dataset: $DATASET"
        echo "Supported: LVIS, COCO, HierText, DocLayNet, HumanRef, Dense200, IC15, M6Doc, RefCOCOg_test, RefCOCOg_val, SROIE, TotalText, VisDrone, FSCD_test"
        exit 1;;
esac

# Apply overrides
SELECTED_JSONL="${TEST_JSONL_PATH:-$DEFAULT_JSONL}"
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"

OUTPUT_DIR="${OUTPUT_DIR}/${GENERATION_MODE}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="$OUTPUT_DIR/answer.jsonl"
EVAL_JSON="$OUTPUT_DIR/eval_results.json"
LOG_FILE="$OUTPUT_DIR/evaluation_log_${TIMESTAMP}.txt"

# ==================== NCCL ====================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ==================== Print Configuration ====================
script_name=$(basename "${BASH_SOURCE[0]}")

echo "=========================================="
echo "=== LocateAnything Grounding Evaluation ==="
echo "=========================================="
echo "DATASET: $DATASET | EVAL_TYPE: $EVAL_TYPE"
echo "NNODES: $NNODES | GPUS: $GPUS | TOTAL: $TOTAL_GPUS"
echo "NODE_RANK: $NODE_RANK | MASTER: $MASTER_ADDR:$PORT"
echo "MODEL_PATH: $MODEL_PATH"
echo "GENERATION_MODE: $GENERATION_MODE"
echo "TEST_JSONL: $SELECTED_JSONL"
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
    "$EVAL_DIR/inference_grounding_ddp.py" \
    --world_size $TOTAL_GPUS \
    --num_nodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $PORT \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$SELECTED_JSONL" \
    --image_root_dir "$IMAGE_ROOT_DIR" \
    --save_path "$SAVE_PATH" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_workers $NUM_WORKERS \
    --eval_type "$EVAL_TYPE" \
    --generation_mode "$GENERATION_MODE" \
    2>&1 | tee -a "$LOG_FILE"

echo "Inference completed. Results: $SAVE_PATH"

# ==================== Step 2: Metrics Evaluation ====================
echo ""
echo "Step 2: Running metrics evaluation..."
echo "=========================================="

python "$EVAL_DIR/metrics/other_metric.py" \
    --data_path "$SAVE_PATH" \
    --output_path "$EVAL_JSON"

echo "Evaluation completed."

# ==================== Step 3: Speed Analysis (BPS/TPS) ====================
echo ""
echo "Step 3: Analyzing speed (TPS, BPS)..."

python "$EVAL_DIR/metrics/analyze_speed.py" \
    --log_file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "$DATASET Evaluation Pipeline completed!"
echo "=========================================="
echo "  Predictions: $SAVE_PATH"
echo "  Eval JSON:   $EVAL_JSON"
echo "  Log:         $LOG_FILE"
echo "=========================================="
