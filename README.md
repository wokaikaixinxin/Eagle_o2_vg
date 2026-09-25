<div align="center">

# A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing -- O2-VG-VLM

</div>


## 1.Install

[AutoDL](https://www.autodl.com/home)

GPU: A single RTX PRO 6000 (96G) * 1

Mirror:

``` shell
Pytorch / version 2.12.1 / python 3.12 (ubuntu22.04) / CUDA 13.0
```

1.1 Install torch

```shell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

1.2 Install flash attention

```shell
pip install flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**Note: magi attention does not support RTX PRO 6000 on 2026.7! Only use flash attention!**

1.3 Install eagle

```shell
cd /root/autodl-tmp
git clone https://github.com/wokaikaixinxin/Eagle_o2_vg.git
cd /root/autodl-tmp/Eagle_o2_vg/Embodied
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2.Data

### 2.1 Download annotations

[locany_recipe](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/locany_recipe)


```shell
cd /root/autodl-tmp
modelscope download --model wokaikaixinxin/LocateAnything-3B-O2-VG --include 'locany_recipe/**' --local_dir ./
```

```
/root/autodl-tmp
├── locany_recipe
│   ├── dior_r_rsvg_rotated
│   │   ├── dior_r_rsvg_rotated_locany_recipe.json
│   │   ├── dior_r_rsvg_rotated_test.jsonl
│   │   ├── dior_r_rsvg_rotated_train.jsonl
│   ├── dior_r_rsvg_rotated_with_universal_obb
│   │   ├── dior_r_rsvg_rotated_locany_recipe.json
│   │   ├── dior_r_rsvg_rotated_test_with_universal_obb.jsonl
│   │   ├── dior_r_rsvg_rotated_train_with_universal_obb.jsonl
│   ├── vrsbench_rotated
│   │   ├── vrsbench_rotated_locany_recipe.json
│   │   ├── vrsbench_rotated_train.jsonl
│   │   ├── vrsbench_rotated_val.jsonl
│   ├── vrsbench_rotated_with_universal_obb
│   │   ├── vrsbench_rotated_locany_recipe.json
│   │   ├── vrsbench_rotated_train_with_universal_obb.jsonl
│   │   ├── vrsbench_rotated_val_with_universal_obb.jsonl
│   ├── avvg_1024x576_rotated
│   │   ├── avvg_rotated_locany_recipe.json
│   │   ├── avvg_rotated_test.jsonl
│   │   ├── avvg_rotated_train.jsonl
│   ├── avvg_rotated_with_universal_obb
│   │   ├── avvg_rotated_locany_recipe.json
│   │   ├── avvg_rotated_test_with_universal_obb.jsonl
│   │   ├── avvg_rotated_train_with_universal_obb.jsonl
```

### 2.2 Download DIOR-R-RSVG

DIOR-R-RSVG [github repo](https://github.com/wokaikaixinxin/DIOR-R-RSVG)

DIOR-R-RSVG [modelscope]()

### 2.3 Download VRSBench

VRSBench [github repo](https://github.com/lx709/VRSBench)

VRSBench [hugging face](https://huggingface.co/datasets/xiang709/VRSBench)


### 2.4 Download AVVG

GeoGround AVVG [github repo](https://github.com/VisionXLab/GeoGround)

GeoGround AVVG [hugging face](https://huggingface.co/datasets/erenzhou/refGeo)

resize AVVG command

```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied
python tools/resize_avvg.py
```

```
/root/autodl-tmp
├── dior_rsvg
│   ├── JPEGImages
├── VRSBench
│   ├── Images_train
│   ├── Images_val
├── avvg_resized_1024x576
│   ├── images
├── refGeo
│   ├── images
│   │   ├── avvg
│   ├── metainfo
```



## 3.Download LocateAnything-3B

```shell
cd /root/autodl-tmp
modelscope download --model wokaikaixinxin/LocateAnything-3B --local_dir ./LocateAnything-3B
```

## 4.VRSBench

### 4.1 Train VRSBench without Universal Oriented Proposals


```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied


LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/vrsbench_rotated/vrsbench_rotated_locany_recipe.json" \
  --output_dir work_dirs/vrsbench_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 300 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 30 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/vrsbench_lr4e-5_grad_acc16_ep2/training_log.txt
```


### 4.2 Test VRSBench without Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/vrsbench_lr4e-5_grad_acc16_ep2/ --annotation /root/autodl-tmp/locany_recipe/vrsbench_rotated/vrsbench_rotated_val.jsonl --image-root /root/autodl-tmp/VRSBench/Images_val --output work_dirs/vrsbench_lr4e-5_grad_acc16_ep2/eval_vrsbench_val.jsonl --generation-mode hybrid --iou-type rotated
```

vrsbench_lr4e-5_grad_acc16_ep2 
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: | 
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/resolve/master/vrsbench_lr4e-5_grad_acc16_ep2/training_log.txt) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_lr4e-5_grad_acc16_ep2/runs) |  [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/vrsbench_lr4e-5_grad_acc16_ep2) |



### 4.3 Train VRSBench with Universal Oriented Proposals

```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied


LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/vrsbench_rotated_with_universal_obb/vrsbench_rotated_locany_recipe.json" \
  --output_dir work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 300 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 30 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/training_log.txt
```


### 4.4 Test VRSBench with Universal Oriented Proposals

```shell
modelscope download --model wokaikaixinxin/LocateAnything-3B-O2-VG --include 'vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/**' --local_dir /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs
```

```shell
python evaluation/eval_rotated_grounding.py --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2 --annotation /root/autodl-tmp/locany_recipe/vrsbench_rotated_with_universal_obb/vrsbench_rotated_val_with_universal_obb.jsonl --image-root /root/autodl-tmp/VRSBench/Images_val --output work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/eval_vrsbench_val.jsonl --generation-mode hybrid --iou-type rotated
```


vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: |
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2) |





## 5.DIOR-R-RSVG

### 5.1 Train DIOR-R-RSVG without Universal Oriented Proposals

```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied


LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/dior_r_rsvg_rotated/dior_r_rsvg_rotated_locany_recipe.json" \
  --output_dir work_dirs/dior_r_rsvg_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 500 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 50 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/dior_r_rsvg_lr4e-5_grad_acc16_ep2/training_log.txt
```


### 5.2 Test DIOR-R-RSVG without Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py \
  --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/dior_r_rsvg_lr4e-5_grad_acc16_ep2 \
  --annotation /root/autodl-tmp/locany_recipe/dior_r_rsvg_rotated/dior_r_rsvg_rotated_test.jsonl \
  --image-root /root/autodl-tmp/dior_rsvg/JPEGImages \
  --output work_dirs/dior_r_rsvg_lr4e-5_grad_acc16_ep2/eval_dior_r_rsvg_test.jsonl \
  --generation-mode hybrid \
  --iou-type rotated
```

dior_r_rsvg_lr4e-5_grad_acc16_ep2
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: |
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/dior_r_rsvg_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/dior_r_rsvg_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/dior_r_rsvg_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/dior_r_rsvg_lr4e-5_grad_acc16_ep2) |

### 5.3 Train DIOR-R-RSVG with Universal Oriented Proposals

```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied


LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/dior_r_rsvg_rotated_with_universal_obb/dior_r_rsvg_rotated_locany_recipe.json" \
  --output_dir work_dirs/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 500 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 50 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2/training_log.txt
```


### 5.4 Test DIOR-R-RSVG with Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py \
  --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2 \
  --annotation /root/autodl-tmp/locany_recipe/dior_r_rsvg_rotated_with_universal_obb/dior_r_rsvg_rotated_test_with_universal_obb.jsonl \
  --image-root /root/autodl-tmp/dior_rsvg/JPEGImages \
  --output work_dirs/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2/eval_dior_r_rsvg_test.jsonl \
  --generation-mode hybrid \
  --iou-type rotated
```


dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: |
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/dior_r_rsvg_with_universal_obb_lr4e-5_grad_acc16_ep2) |


## 6. AVVG

### 6.1 Train AVVG without Universal Oriented Proposals

```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied

LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/avvg_1024x576_rotated/avvg_rotated_locany_recipe.json" \
  --output_dir work_dirs/avvg_1024x576_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 300 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 30 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/avvg_1024x576_lr4e-5_grad_acc16_ep2/training_log.txt
```

### 6.2 Test AVVG without Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py \
  --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/avvg_1024x576_lr4e-5_grad_acc16_ep2 \
  --annotation /root/autodl-tmp/locany_recipe/avvg_1024x576_rotated/avvg_rotated_test.jsonl \
  --image-root /root/autodl-tmp/avvg_resized_1024x576/images \
  --output work_dirs/avvg_1024x576_lr4e-5_grad_acc16_ep2/eval_avvg_1024x576_test.jsonl \
  --generation-mode hybrid \
  --iou-type rotated
```


avvg_1024x576_lr4e-5_grad_acc16_ep2
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: |
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/avvg_1024x576_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/avvg_1024x576_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/avvg_1024x576_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/avvg_1024x576_lr4e-5_grad_acc16_ep2) |


### 6.3 Train AVVG with Universal Oriented Proposals


```shell
cd /root/autodl-tmp/Eagle_o2_vg/Embodied

LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /root/autodl-tmp/LocateAnything-3B \
  --meta_path "/root/autodl-tmp/locany_recipe/avvg_rotated_with_universal_obb/avvg_rotated_locany_recipe.json" \
  --output_dir work_dirs/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2 \
  --overwrite_output_dir False \
  --max_steps 300 \
  --block_size 7 \
  --box_coord_dim 5 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_backbone False \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_steps 30 \
  --lr_scheduler_type "cosine" \
  --max_grad_norm 1.0 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_every_n_hours 0 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --packing_buffer_size 32 \
  --max_seq_length 8192 \
  --max_num_tokens_per_sample 8192 \
  --max_num_tokens 8192 \
  --grad_checkpoint True \
  --group_by_length False \
  --optim adamw_torch \
  --report_to "tensorboard" \
  --mlp_connector_layers 2 \
  --do_train True \
  2>&1 | tee -a work_dirs/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2/training_log.txt
```

### 6.4 Test AVVG with Universal Oriented Proposals


```shell
python evaluation/eval_rotated_grounding.py \
  --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2 \
  --annotation /root/autodl-tmp/locany_recipe/avvg_rotated_with_universal_obb/avvg_rotated_test_with_universal_obb.jsonl \
  --image-root /root/autodl-tmp/avvg_resized_1024x576/images \
  --output work_dirs/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2/eval_avvg_1024x576_test.jsonl \
  --generation-mode hybrid \
  --iou-type rotated
```



avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2
| model checkPoint | training logs | tensorboard | test result |
| :------: | :--: | :-----: | :------: |
| [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/avvg_1024x576_with_universal_obb_lr4e-5_grad_acc16_ep2) |


## Citation




