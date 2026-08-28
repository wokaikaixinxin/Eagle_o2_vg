<div align="center">

# 🦅 Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models

<p>
    <img src="./assets/VideoMME.png" alt="Eagle" width="500" height="auto">
</p>

</div>

Eagle 2.5 is a family of frontier vision-language models (VLMs) designed for long-context multimodal learning. While most existing VLMs focus on short-context tasks, Eagle 2.5 addresses the challenges of long video comprehension and high-resolution image understanding, providing a generalist framework for both. Eagle 2.5 supports up to 512 video frames and is trained jointly on image + video data.

This repository provides the end-to-end guidance and scripts for the environment setup, data preparation, training, and inference of the Eagle VLM.



## Highlight

### 🚀 Strong Results Across The Board

- SOTA on 6 out of 10 long video benchmarks
- Outperforms GPT-4o (0806) on 3/5 video tasks
- Outperforms Gemini 1.5 Pro on 4/6 video tasks
- Matches or outperforms Qwen2.5-VL-72B on multiple key datasets
- 72.4% on Video-MME with 512 input frames
- Strong image understanding with consistent improvement over Eagle 2, matching Qwen2.5-VL.

### 🎯 Key Innovations

- **Information-First Sampling**:
  - *Image Area Preservation (IAP)*: Optimizes image tiling to retain most of the original image area and aspect ratio, preserving fine-grained details.
  - *Automatic Degrade Sampling (ADS)*: Dynamically balances visual and textual input, ensuring complete text retention while maximizing visual content within context length constraints.
- **Progressive Mixed Post-Training**:
  - Gradually increases context length during training, enhancing the model's ability to process varying input sizes and improving information density over static sampling.
- **Diversity-Driven Data Recipe**:
  - Combines open-source data (human-annotated and synthetic) with the self-curated Eagle-Video-110K dataset, collected via a diversity-driven strategy and annotated with both story-level and clip-level QA pairs.

### ⚡ Efficiency & Framework Optimization

- **GPU Memory Optimization**:
  - Integrate Triton-based fused operators replacing PyTorch’s MLP, RMSNorm, and RoPE implementations.
  - Reduced GPU memory with fused linear layers + cross-entropy loss (removes intermediate logit storage) and CPU-offloading of hidden states.
  - Sufficient to fit up to 32K context length with an 8B model on a single GPU. 
- **Distributed Context Parallelism**:
  - Adopts a two-layer communication group based on Ulysses and Ring/Context Parallelism building on USP.
  - Implements ZigZag Llama3-style Context Parallelism with all-gather KV to reduce communication latency.
- **Video Decoding Acceleration**:
  - Optimized sparse video frame sampling with rapid video metadata parsing, improved long video decoding and reduced memory consumption.
- **Inference Acceleration**:
  - Supports vLLM deployment with reduced memory and accelerated inference.
  

## Model Details

- **Model Type**: Long-context vision-language model
- **Architecture**: 
  - Vision encoder: Siglip2-So400m-Patch16-512
  - Language model: Qwen2.5-7B-Instruct
  - Multimodal base architecture: LLaVA with tiling-based vision input
- **Supported Inputs**: 
  - Long video sequences (up to 512 frames)
  - High-resolution images (up to 4K HD input size)
  - Multi-page documents
  - Long text
- **Training Strategy**: 
  - Progressive mixed post-training, expanding from 32K to 128K context length
  - Information-first sampling for optimal visual and textual information retention
- **Training Data**: 
  - Open-source video and document datasets
  - Eagle-Video-110K (110K long videos with dual-level annotation)


## Getting Started

### 📚 Onboarding

Recommended order:

1) Set environment variables → 2) Install → 3) Prepare data → 4) Train → 5) Demo → 6) Inference

- Onboarding overview: see `./document/0.onboarding.md`

### ⚙️ Installation & Environment

- Detailed steps and dependencies: `./document/1.installing.md`
  - Conda environment (Python 3.10)
  - PyTorch and FlashAttention (match your CUDA)
  - Install this repo with `pip install -e .`
  - Troubleshooting notes (specific Transformers version, OpenCV dependencies, etc.)

### 📂 Data Preparation (Playground)

- Directory structure and JSONL/LMDB examples: `./document/2.preparing_playground.md`
  - `playground/sft_recipe` (data recipe)
  - `playground/sft_jsonl` and `playground/sft_data` (annotations and raw data)
  - Example parquet→LMDB conversion scripts are not included in this repo
  - Use `shell/prepare.sh` to normalize and generate `.prepare.json` (internal `submit_prepare_job.sh` is not included)
  - LMDB reading example and tips: `./document/how_to_use_lmdb_to_read_images.md`

### 💪 Training (Stage-2 / Finetuning)

- Full training entry points and multinode/multigpu options: `./document/3.training.md`
  - Single-node example: `GPUS=8 bash shell/train_stage2.sh 1 work_dirs/eagle2.5_debug`
  - Multi-node example (srun/internal submit_job): `PARTITION=xxx GPUS=16 bash shell/train_stage2.sh 2 work_dirs/eagle2.5_multinode`

### ✨ Launching Streamlit Demo

- Interactive testing of the VLM with UI. Refer to document for more details: `./document/4.streamlit_demo.md`

### 🔮 Inference

- End-to-end usage and multimodal examples (single/multiple images, single/multiple videos, streaming, batch): `./document/5.inference.md`
  - Load with `transformers` `AutoModel`/`AutoProcessor`: `"nvidia/Eagle-2.5-8B"`
  - Recommended `torch_dtype=torch.bfloat16`; run `model.generate(...)` on GPU


## Benchmark Results

### 🎥 Video Benchmarks

| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| MVBench<sub>test</sub>                     | -                  | -                 | 72.0                | 69.6                | 74.8                |
| Perception_test<sub>val</sub>              | -                  | -                 | -                   | 70.5                | 82.0                |
| EgoSchema<sub>fullset</sub>                | -                  | 72.2              | -                   | 65.0                | 72.2                |
| MMB-Video                                  | 1.63               | 1.30              | 1.68                | 1.79                | 1.94                |
| MLVU<sub>val</sub>                         | -                  | -                 | 68.9                | 70.2                | 77.6                |
| LVBench<sub>val</sub>                      | 66.7               | 64.0              | 60.0                | 56.0                | 66.4                |
| Video-MME<sub>w/o subtitle</sub>           | 71.9               | 75.0              | 64.2                | 65.1                | 72.4                |
| Video-MME<sub>w subtitle</sub>             | 77.2               | 81.3              | 66.9                | 71.6                | 75.7                |
| CG-Bench<sub>Clue</sub>                    | 58.6               | 50.9              | -                   | 44.5                | 55.8                |
| CG-Bench<sub>Long</sub>                    | 44.9               | 37.8              | -                   | 35.5                | 46.6                |
| CG-Bench<sub>mIoU</sub>                    | 5.73               | 3.85              | -                   | 2.48                | 13.4                |
| HourVideo<sub>Dev</sub>                    | -                  | 37.2              | -                   | -                   | 44.5                |
| HourVideo<sub>Test</sub>                   | -                  | 37.4              | -                   | -                   | 41.8                |
| Charade-STA<sub>mIoU</sub>                 | 35.7               | -                 | -                   | 43.6                | 65.9                |
| HD-EPIC                                    | -                  | 37.6              | -                   | -                   | 42.9                |
| HRVideoBench                               | -                  | -                 | -                   | -                   | 68.5                |
| EgoPlan<sub>val</sub>                      | -                  | -                 | -                   | -                   | 45.3                |

### 🖼️ Image Benchmarks

| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| DocVQA<sub>test</sub>                      | 92.8               | 93.1              | 93.0                | 95.7                | 94.1                |
| ChartQA<sub>test</sub>                     | 85.7               | 87.2              | 84.8                | 87.3                | 87.5                |
| InfoVQA<sub>test</sub>                     | 79.2               | 81.0              | 77.6                | 82.6                | 80.4                |
| TextVQA<sub>val</sub>                      | 77.4               | 78.8              | 79.1                | 84.9                | 83.7                |
| OCRBench<sub>test</sub>                    | 736                | 754               | 822                 | 864                 | 869                 |
| MMstar<sub>test</sub>                      | 64.7               | 59.1              | 62.8                | 63.9                | 66.2                |
| RWQA<sub>test</sub>                        | 75.4               | 67.5              | 70.1                | 68.5                | 76.7                |
| AI2D<sub>test</sub>                        | 84.6               | 79.1              | 84.5                | 83.9                | 84.5                |
| MMMU<sub>val</sub>                         | 69.1               | 62.2              | 56.0                | 58.6                | 55.8                |
| MMBench_V11<sub>test</sub>                 | 83.1               | 74.6              | 83.2                | 82.6                | 81.7                |
| MMVet<sub>GPT-4-Turbo</sub>                | 69.1               | 64.0              | 62.8                | 67.1                | 62.9                |
| HallBench<sub>avg</sub>                    | 55.0               | 45.6              | 50.1                | 52.9                | 54.7                |
| MathVista<sub>testmini</sub>               | 63.8               | 63.9              | 64.4                | 68.2                | 67.8                |
| Avg Score                                  | 74.9               | 71.7              | 73.1                | 75.6                | 75.6                |

### 🦾 Embodied Benchmarks
| Benchmark                                  | GPT-4o             | Gemini-1.5 Pro    | InternVL2.5-8B      | Qwen2.5-VL-8B       | **Eagle2.5-8B**     |
|--------------------------------------------|--------------------|-------------------|---------------------|---------------------|---------------------|
| OpenEQA                                    | -                  | -                 | -                   | -                   | 63.5                |
| ERQA                                       | 47.0               | 41.8              | -                   | -                   | 38.3                |
| EgoPlan<sub>val</sub>                      | -                  | -                 | -                   | -                   | 45.3                |


## License

- See [LICENSE](../LICENSE) for the code of this repository.
- See [LICENSE_MODEL](./LICENSE_MODEL) for the models of Eagle 2 and Eagle 2.5.

For detailed parameter explanations and launcher script notes, see: `./document/explain_script_arguments.md`.
