```bash
#!/usr/bin/env bash
set -x

GPUS=${GPUS:-8}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"work_dirs/debug"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TOTAL_GPUS=$((GPUS * NNODES))

TOKEN_PER_SEQUENCE_PARALLEL_GROUP=16384 # decided by packing size
TOKEN_PER_GPU=16384 # SIGLIP + 7B LLM, decided by Vision Encoder and LLM
TOTAL_TOKENS_EXP=${ITER_TOTAL_TOKENS_EXP:-17} #17->128K 18->0.25M, 19->0.5M, 20->1M, 21->2M, 22->4M, 23->8M, 24->16M
TOTAL_TOKENS_PER_ITER=$((2 ** TOTAL_TOKENS_EXP))

SEQUENCE_PARALLEL_SIZE=$((TOKEN_PER_SEQUENCE_PARALLEL_GROUP / TOKEN_PER_GPU))
SEQUENCE_PARALLEL_NUM_GROUP=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))
DATA_PARALLEL_WORLD_SIZE=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))

GRADIENT_ACC=$(($TOTAL_TOKENS_PER_ITER / ($TOKEN_PER_GPU * $TOTAL_GPUS)))
SAMPLE_DIV=1

LOSS_VERSION="default"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
NCCL_DEBUG=INFO

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


export WANDB_PROJECT='Eagle-Next'
export WANDB_LOG_MODEL=False
script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")

LAUNCHER=pytorch torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
  eaglevl/train/eagle_2_5_vl_finetune.py \
  --model_name_or_path "work_dirs/Eagle2-5-VL-8B-Preview" \
  --conv_style "qwen2-chat" \
  --normalize_type "siglip" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "playground/sft_recipe/debug.prepared.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --max_dynamic_tiles 12 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 0 \
  --bf16 True \
  --use_online_packing True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 1 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length ${TOKEN_PER_SEQUENCE_PARALLEL_GROUP} \
  --sequence_parallel_degree ${SEQUENCE_PARALLEL_SIZE} \
  --sample_length_div ${SAMPLE_DIV} \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --deepspeed "deepspeed_configs/zero_stage3_config.json" \
  --loss_version ${LOSS_VERSION} \
  --report_to "tensorboard"\
  --run_name $script_name \
  --use_onelogger True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
```

# Script Argument Explanation

This document explains the parameters and variables used in the provided shell script for launching the `eaglevl/train/eagle_2_5_vl_finetune.py` training script.

## Environment Variables and Shell Variables

*   `GPUS=${GPUS:-8}`: Number of GPUs to use per compute node. Defaults to 8 if the `GPUS` environment variable is not set.
*   `NNODES=${1:-1}`: Total number of compute nodes to use. Takes the value from the first command-line argument (`$1`). Defaults to 1 if no argument is provided.
*   `OUTPUT_DIR=${2:-"work_dirs/debug"}`: Directory to store training outputs (checkpoints, logs, etc.). Takes the value from the second command-line argument (`$2`). Defaults to "work\_dirs/debug" if no argument is provided.
*   `NODE_RANK=${NODE_RANK:-0}`: Global rank (0-based) of the current node in a distributed setup. Defaults to 0 if the `NODE_RANK` environment variable is not set.
*   `PORT=${PORT:-29500}`: Port number used by the master node in distributed training. Defaults to 29500 if the `PORT` environment variable is not set.
*   `MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}`: IP address or hostname of the master node in distributed training. Defaults to "127.0.0.1" (localhost) if the `MASTER_ADDR` environment variable is not set.
*   `NCCL_DEBUG=INFO`: Sets the debug level for NVIDIA NCCL (NVIDIA Collective Communications Library) to INFO, showing information about communication operations.
*   `WANDB_PROJECT='Eagle-Next'`: Specifies the project name for Weights & Biases (W&B) logging.
*   `WANDB_LOG_MODEL=False`: Configures W&B not to log model files automatically.
*   `script_path=${BASH_SOURCE[0]}`: Gets the path of the current script itself.
*   `script_name=$(basename "$script_path")`: Extracts the filename of the script from its path.

## Calculated Variables

*   `TOTAL_GPUS=$((GPUS * NNODES))`: Calculates the total number of GPUs used for training.
*   `TOKEN_PER_SEQUENCE_PARALLEL_GROUP=16384`: Number of tokens processed per sequence parallel group. This is often determined by the model's packing size.
*   `TOKEN_PER_GPU=16384`: Number of tokens processed per GPU. The comment suggests this depends on the Vision Encoder and LLM configuration (e.g., SIGLIP + 7B LLM).
*   `TOTAL_TOKENS_EXP=${ITER_TOTAL_TOKENS_EXP:-17}`: Exponent used to calculate the total tokens per iteration. The comment provides examples (17 -> 128K, 18 -> 0.25M, etc.). Defaults to 17 if the `ITER_TOTAL_TOKENS_EXP` environment variable is not set.
*   `TOTAL_TOKENS_PER_ITER=$((2 ** TOTAL_TOKENS_EXP))`: Calculates the total number of tokens processed per iteration.
*   `SEQUENCE_PARALLEL_SIZE=$((TOKEN_PER_SEQUENCE_PARALLEL_GROUP / TOKEN_PER_GPU))`: Calculates the sequence parallelism size (number of GPUs per sequence parallel group).
*   `SEQUENCE_PARALLEL_NUM_GROUP=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))`: Calculates the number of sequence parallel groups.
*   `DATA_PARALLEL_WORLD_SIZE=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))`: Calculates the data parallel world size (equivalent to the number of sequence parallel groups).
*   `GRADIENT_ACC=$(($TOTAL_TOKENS_PER_ITER / ($TOKEN_PER_GPU * $TOTAL_GPUS)))`: Calculates the number of gradient accumulation steps. This simulates a larger batch size with limited GPU memory.
*   `SAMPLE_DIV=1`: A factor for sample length division (its specific use depends on the training script's internal logic).
*   `LOSS_VERSION="default"`: Specifies the version of the loss function to use.

## `torchrun` Launcher Arguments

*   `LAUNCHER=pytorch`: Specifies the PyTorch launcher (somewhat redundant as the command is `torchrun`).
*   `--nnodes=$NNODES`: Passes the total number of nodes to `torchrun`.
*   `--node_rank=$NODE_RANK`: Passes the current node's rank to `torchrun`.
*   `--master_addr=$MASTER_ADDR`: Passes the master node's address to `torchrun`.
*   `--nproc_per_node=$GPUS`: Passes the number of processes per node (equal to GPU count) to `torchrun`.
*   `--master_port=$PORT`: Passes the master node's port to `torchrun`.

## `eaglevl/train/eagle_2_5_vl_finetune.py` Script Arguments

*   `--model_name_or_path "work_dirs/Eagle2-5-VL-8B-Preview"`: Specifies the pre-trained model to load or its path.
*   `--conv_style "qwen2-chat"`: Specifies the conversation format style (e.g., template for prompts and responses).
*   `--normalize_type "siglip"`: Specifies the image normalization type, using SigLIP's method here.
*   `--output_dir ${OUTPUT_DIR}`: Specifies the training output directory (passes the previously set shell variable).
*   `--meta_path "playground/sft_recipe/debug.prepared.json"`: Path to the JSON file containing training data metadata (file paths, sample info, etc.).
*   `--overwrite_output_dir False`: If the output directory exists, do not overwrite it.
*   `--force_image_size 448`: Forces all images to be resized to 448x448.
*   `--max_dynamic_tiles 12`: Maximum number of tiles allowed when using dynamic image sizes.
*   `--down_sample_ratio 0.5`: Downsampling ratio of pixel shuffle.
*   `--pad2square False`: Do not pad images to make them square.
*   `--freeze_llm True`: Freezes the weights of the Large Language Model (LLM) part, preventing updates during fine-tuning.
*   `--freeze_mlp False`: Does not freeze the Multi-Layer Perceptron (MLP) weights.
*   `--freeze_backbone False`: Does not freeze the visual backbone weights.
*   `--vision_select_layer -1`: Specifies which layer(s) features to select from the vision encoder (-1 usually means the last layer).
*   `--use_data_resampling False`: Whether to use data resampling.
*   `--dataloader_num_workers 0`: Number of subprocesses for data loading. 0 means loading data in the main process.
*   `--bf16 True`: Use bfloat16 mixed-precision for training to save memory and potentially speed up training.
*   `--use_online_packing True`: **Dynamically pack short sequences into longer ones during training to improve efficiency**.
*   `--num_train_epochs 1`: Total number of training epochs.
*   `--per_device_train_batch_size 1`: Training batch size per GPU. The effective batch size is `per_device_train_batch_size * TOTAL_GPUS * gradient_accumulation_steps`.
*   `--gradient_accumulation_steps ${GRADIENT_ACC}`: Number of gradient accumulation steps (passes the previously calculated shell variable).
*   `--save_strategy "steps"`: Model saving strategy, save based on steps.
*   `--save_steps 100`: Save a model checkpoint every N steps.
*   `--save_total_limit 3`: Maximum number of checkpoints to keep.
*   `--learning_rate 4e-5`: Learning rate for training.
*   `--weight_decay 0.05`: Coefficient for weight decay (L2 regularization).
*   `--warmup_ratio 0.03`: Proportion of total training steps used for the learning rate warmup phase.
*   `--lr_scheduler_type "cosine"`: Type of learning rate scheduler, using cosine annealing here.
*   `--logging_steps 1`: Log metrics (e.g., training loss) every N steps.
*   `--max_seq_length ${TOKEN_PER_SEQUENCE_PARALLEL_GROUP}`: Maximum sequence length the model can process (passes the previously set shell variable).
*   `--sequence_parallel_degree ${SEQUENCE_PARALLEL_SIZE}`: Degree of sequence parallelism (passes the previously calculated shell variable).
*   `--sample_length_div ${SAMPLE_DIV}`: Sample length division factor (passes the previously set shell variable).
*   `--do_train True`: Execute the training process.
*   `--grad_checkpoint True`: Use gradient checkpointing to save memory by not storing intermediate activations during the forward pass and recomputing them during the backward pass.
*   `--group_by_length True`: Group samples of similar lengths into the same batch to reduce padding and improve efficiency. useless while setting online packing
*   `--dynamic_image_size True`: Allow dynamic image sizes during training instead of forcing a fixed size.
*   `--use_thumbnail True`: Potentially indicates using image thumbnails as low-resolution inputs.
*   `--deepspeed "deepspeed_configs/zero_stage3_config.json"`: Path to the DeepSpeed configuration file, enabling DeepSpeed optimizations (like ZeRO).
*   `--loss_version ${LOSS_VERSION}`: Specifies the loss function version (passes the previously set shell variable).
*   `--report_to "tensorboard"`: Report training metrics to TensorBoard.
*   `--run_name $script_name`: Sets the run name in W&B or TensorBoard (uses the script's filename).
*   `--use_onelogger True`: Potentially indicates using a specific logging library or configuration. (***NVIDIA internel user only**)
*   `--save_every_n_hours 4` \ (***NVIDIA internel user only**) every job only runs 2 or 4 hours, so that we need save the checkpoint in the tile of every N hours.

## Command Redirection

*   `2>&1`: Redirects standard error (stderr, file descriptor 2) to standard output (stdout, file descriptor 1).
*   `| tee -a "${OUTPUT_DIR}/training_log.txt"`: Pipes the combined stdout and stderr to the `tee` command. `tee` does two things:
    *   Displays the output on the terminal (stdout).
    *   Appends (`-a`) the output to the file `${OUTPUT_DIR}/training_log.txt`.

This setup allows you to see the training log on the screen while also saving a complete copy to a log file.