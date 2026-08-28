<div align="center">

#  Oriented Object Visual Grounding in Remote Sensing Images

</div>


## 1.Data

### 1.1 Download annotations

[locany_recipe](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/locany_recipe)

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

### 1.2 Download DIOR-R-RSVG

DIOR-R-RSVG [github repo](https://github.com/wokaikaixinxin/DIOR-R-RSVG)

DIOR-R-RSVG [modelscope]()

### 1.3 Download VRSBench

VRSBench [github repo](https://github.com/lx709/VRSBench)

VRSBench [hugging face](https://huggingface.co/datasets/xiang709/VRSBench)

### 1.4 Download AVVG

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

## 2.Install

[AutoDL](https://www.autodl.com/home)

GPU:

A single RTX PRO 6000 (96G) * 1

Mirror:

``` shell
Pytorch / version 2.12.1 / python 3.12 (ubuntu22.04) / CUDA 13.0
```

Install torch

```shell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

Install flash attention

```shell
pip install flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**Note: magi attention does not support RTX PRO 6000 on 2026.7! Only use flash attention!**

**Note: magi attention does not support RTX PRO 6000 on 2026.7! Only use flash attention!**

Install eagle

```shell
cd /root/autodl-tmp
git clone https://github.com/wokaikaixinxin/Eagle_o2_vg.git
cd /root/autodl-tmp/Eagle_o2_vg/Embodied
pip install -e . -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3.Download LocateAnything-3B

```shell
cd /root/autodl-tmp
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download nvidia/LocateAnything-3B --local-dir ./
```

## 4.VRSBench

### Train VRSBench without Universal Oriented Proposals


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


### Test VRSBench without Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/vrsbench_lr4e-5_grad_acc16_ep2/ --annotation /root/autodl-tmp/locany_recipe/vrsbench_rotated/vrsbench_rotated_val.jsonl --image-root /root/autodl-tmp/VRSBench/Images_val --output work_dirs/vrsbench_lr4e-5_grad_acc16_ep2/eval_vrsbench_val.jsonl --generation-mode hybrid --iou-type rotated
```

| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: | 
|vrsbench_lr4e-5_grad_acc16_ep2 | [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/resolve/master/vrsbench_lr4e-5_grad_acc16_ep2/training_log.txt) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_lr4e-5_grad_acc16_ep2/runs) |  [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/vrsbench_lr4e-5_grad_acc16_ep2) |



### Train VRSBench with Universal Oriented Proposals

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


### Test VRSBench with Universal Oriented Proposals

```shell
python evaluation/eval_rotated_grounding.py --model /root/autodl-tmp/Eagle_o2_vg/Embodied/work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2 --annotation /root/autodl-tmp/locany_recipe/vrsbench_rotated_with_universal_obb/vrsbench_rotated_val_with_universal_obb.jsonl --image-root /root/autodl-tmp/VRSBench/Images_val --output work_dirs/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/eval_vrsbench_val.jsonl --generation-mode hybrid --iou-type rotated
```



| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: |
| vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2 | [model checkPoint](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2) | [training logs](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/file/view/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2%2Ftraining_log.txt?status=1) | [tensorboard](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2/runs) | [test result](https://modelscope.cn/models/wokaikaixinxin/LocateAnything-3B-O2-VG/tree/master/o2_vg_vlm_test_result_different_mode/vrsbench_with_universal_obb_lr4e-5_grad_acc16_ep2) |












| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: |
|vrsbench_lr4e-5_grad_acc16_ep2 | [model checkPoint]() | [training logs]() | [tensorboard]() | [test result]() |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|        |          |        |         |       |         |      |         |      |













| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: |
|vrsbench_lr4e-5_grad_acc16_ep2 | [model checkPoint]() | [training logs]() | [tensorboard]() | [test result]() |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|        |          |        |         |       |         |      |         |      |














| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: |
|vrsbench_lr4e-5_grad_acc16_ep2 | [model checkPoint]() | [training logs]() | [tensorboard]() | [test result]() |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|        |          |        |         |       |         |      |         |      |

















| method | model checkPoint | training logs | tensorboard | test result |
| :----: | :------: | :--: | :-----: | :------: |
|vrsbench_lr4e-5_grad_acc16_ep2 | [model checkPoint]() | [training logs]() | [tensorboard]() | [test result]() |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|        |          |        |         |       |         |      |         |      |


## Citation
If you find this project useful, please consider citing our works:
```latex
@inproceedings{wang2025locateanything,
    title={LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding},
    author={Shihao Wang and Shilong Liu and Yuanguo Kuang and Xinyu Wei and Yangzhou Liu and Zhiqi Li and Yunze Man and Guo Chen and Andrew Tao and Guilin Liu and Jan Kautz and Lei Zhang and Zhiding Yu},
    booktitle={ECCV},
    year={2026}
}
```

```latex
@inproceedings{man2025locateanything3d,
    title   = {LocateAnything3D: Vision-Language 3D Detection with Chain-of-Sight},
    author  = {Yunze Man and Shihao Wang and Guowen Zhang and Johan Bjorck and Zhiqi Li and Liang-Yan Gui and Jim Fan and Jan Kautz and Yu-Xiong Wang and Zhiding Yu},
    booktitle = {CVPR},
    year    = {2026},
}
```

```latex
@inproceedings{chen2025eagle2.5,
    title={Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models},
    author={Guo Chen and Zhiqi Li and Shihao Wang and Jindong Jiang and Yicheng Liu and Lidong Lu and De-An Huang and Wonmin Byeon and Matthieu Le and Max Ehrlich and Tong Lu and Limin Wang and Bryan Catanzaro and Jan Kautz and Andrew Tao and Zhiding Yu and Guilin Liu},
    booktitle={NeurIPS},
    year={2025}
}
```

```latex
@article{li2025eagle2,
    title={Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models}, 
    author={Zhiqi Li and Guo Chen and Shilong Liu and Shihao Wang and Vibashan VS and Yishen Ji and Shiyi Lan and Hao Zhang and Yilin Zhao and Subhashree Radhakrishnan and Nadine Chang and Karan Sapra and Amala Sanjay Deshmukh and Tuomas Rintamaki and Matthieu Le and Ilia Karmanov and Lukas Voegtle and Philipp Fischer and De-An Huang and Timo Roman and Tong Lu and Jose M. Alvarez and Bryan Catanzaro and Jan Kautz and Andrew Tao and Guilin Liu and Zhiding Yu},
    journal={arXiv:2501.14818},
    year={2025}
}
```

```latex
@inproceedings{shi2025eagle,
    title = {Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders}, 
    author={Min Shi and Fuxiao Liu and Shihao Wang and Shijia Liao and Subhashree Radhakrishnan and De-An Huang and Hongxu Yin and Karan Sapra and Yaser Yacoob and Humphrey Shi and Bryan Catanzaro and Andrew Tao and Jan Kautz and Zhiding Yu and Guilin Liu},
    booktitle={ICLR},
    year={2025}
}
```


## License/Terms of Use
- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file. Portions of the code in this repo are reused and subject to their original licenses. Some files have been modified, with appropriate attribution and additional license headers added where applicable.
- The pretrained model weights are released under either the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/deed.en) or the [NVIDIA License](./Eagle2_5/LICENSE_MODEL). The models are research preview intended for non-commercial use only.
- Eagle models are improved using Qwen.
- For code contributions to Eagle, please refer to the [Contribution Guide](CONTRIBUTING.md).
- Users are reminded to ensure that their use of the dataset and model weights is in compliance with all applicable laws and regulations.


## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-HR](https://github.com/luogen1996/LLaVA-HR) and [InternVL](https://github.com/OpenGVLab/InternVL): The Eagle codebase has integrated modified components from these repositories. Many thanks for the great open-source projects.
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): We use derivatives of these repositories for evaluation. Many thanks for the wonderful tools.
- Thanks to [Cambrian](https://cambrian-mllm.github.io), [LLaVA-One-Vision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/), [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) and many other works for the great efforts in open-sourcing data.
- The team would like to give special thanks to the NVIDIA TSE Team, including Chen Fu, Yuchao Jin, Le An, and Josh Park, for their exceptional work on the optimized TensorRT and edge deployment of Eagle.
