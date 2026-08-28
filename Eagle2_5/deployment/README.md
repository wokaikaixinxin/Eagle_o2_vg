# Eagle2.5 Deployment with TensorRT and TensorRT-LLM on x86 Platforms

This document demonstrates the deployment of the [Eagle-2.5](https://arxiv.org/pdf/2504.15271) utilizing [TensorRT](https://github.com/NVIDIA/TensorRT) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). In this deployment demo we will provide a step-by-step overview of the deployment process, including the overall strategy, environment setup, engine build, engine benchmarking and result analysis.

## Table of Contents

1. [Deployment strategy](#strategy)
2. [Environment setup](#env)
3. [Vision engine build](#vision)
4. [LLM engine build](#llm)
5. [Eagle2.5 Inference with TensorRT-LLM and TensorRT ](#inference)


## Deployment strategy <a name="strategy"></a>
To enhence inference efficiency of Eagle2.5, engines are built seperately for the vision component and the LLM component. Below are the pipelines for deploying the two components:
 - The vision component: 
    1) export ONNX model
    2) build engines with TensorRT
 - The LLM component:
    1) convert the checkpoints to Hugging Face safetensor with TensorRT-LLM
    2) build engines with `trtllm-build`

We use TensorRT 10.11 public docker, and TensorRT-LLM 0.20 to deploy the Eagle2.5 on X86_64 Linux platforms. (see this [report](https://arxiv.org/pdf/2504.15271) for model details.) Please notice that to run TensorRT-LLM within our environment, a patch must be applied to the TensorRT-LLM. You may refer to [Environment setup](#env) section for more details.

## Environmnet setup <a name="env"></a>
Please ensure that the folder structure complies with the requirements outlined in this [README](../README.md). You will need to prepare the following:
- The model checkpoints
- TensorRT-LLM ([commit](https://github.com/NVIDIA/TensorRT-LLM/tree/7c828d767f12dd505187e626357a80da9f05a99a))

We recommend starting with the [Dockerfile](./setup_x86.dockerfile):
```bash
docker build -t eagle2_5-deploy:v0 --file ./setup_x86.dockerfile .
docker run -it --name eagle2_5-deploy --gpus all --shm-size=8g -v <workspace>:<workspace> eagle2_5-deploy:v0 /bin/bash
```

To setup TensorRT library, you may refer to [TensorRT github](https://github.com/NVIDIA/TensorRT). You should choose the correct wheel according to the python version in your environment. In this demo, we use the default TensorRT 10.11.0.33 with Python 3.12.3 in the [tensorrt:25.06-py3](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html#rel-25-06) docker release.

We also need to build the TensorRT-LLM wheels within the Docker environment. Please follow the commands to build and install the TensorRT-LLM wheels in the docker.
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 7c828d767f12dd505187e626357a80da9f05a99a
git submodule update --init --recursive
apt update && apt install git-lfs
git lfs install && git lfs pull
# This patch is experimental, and targeting on Eagle2.5 llm inference only.
git apply ../tensorrt_llm.patch

# Please change CUDA_ARCH according to your own hardware spec
export CUDA_ARCH=<Choose according to your hardware>
python3 ./scripts/build_wheel.py --cuda_architectures=$CUDA_ARCH -D ENABLE_MULTI_DEVICE=0 --job_count=8 --clean --benchmarks

# Then install tensorrt_llm alone with some other dependencies
pip3 install ./build/tensorrt_llm_*.whl --force-reinstall --no-deps

```

## Vision engine build <a name="vision"></a>

The vision component of Eagle2.5 generates vision embeddings for the LLM model as inputs. We export the vision moodel to a unified ONNX model and build engines based on the ONNX model subsequently. In the following example, we provides a unified script to export ONNX file and command line to generate TensorRT engines in FP16 precision. 

```bash
export PYTHONPATH="$(pwd)/.."
python export_vision_onnx.py --config_path=<config_path> --onnx_file=<save_onnx_pth> --model_path=<model_path> 
/opt/tensorrt/bin/trtexec --onnx_file=<save_onnx_path> --saveEngine=<save_trt_engine_path> --minShapes=<1xHxWx3> --optShapes=<OxHxWx3> --maxShapes=<MxHxWx3> --fp16 --useCudaGraph 

```
By running the above commands, you will obtain both the ONNX file and TRT engine for the vision component of Eagle2.5. Please refer to your applications to determine `O`, `H`, `W` and `M`.


## LLM engine build <a name="llm"></a>
For the LLM part of the Eagle2.5 model, we provides guidance for the following precisions as examples, FP16 and INT4 weight-only quantization. To export the LLM part of the Eagle2.5 model, please use the `export_llm_engine.py` scripts, which will convert the Hugging Face checkpoint to Hugging Face safetensors format using apis from TensorRT-LLM. We are adopting the Qwen model export python script from [TensorrRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/7c828d767f12dd505187e626357a80da9f05a99a/examples/models/core/qwen/convert_checkpoint.py) as the LLM part of Eagle2.5 shares similar network archiecture comparing to Qwen model. To export the LLM engine in FP16 precision:
```bash
# FP16 engine
python export_llm_engine.py --model_dir <model_path> --dtype auto --output_dir <llm_engine_path>/x86/fp16/
```
We also provide a sample command if you would like to export the LLM part in INT4 weight-only quantization format:
```bash
# FP16 activation, INT4 weight (w4a16 weight-only quantization)
python export_llm_engine.py --model_dir <model_path> --dtype auto --output_dir <llm_engine_path>/x86/int4/ --use_weight_only --weight_only_precision int4
```
After this step, the `.safetensors` file should be generated. To build the TensorRT-LLM engine, we can use the `trtllm-build` command line as follows:
```bash
# FP16 engine and FP16 activation, INT4 
trtllm-build --checkpoint_dir <llm_engine_path>/x86/<fp16_or_int4> --output_dir <llm_engine_path>/x86/<fp16_or_int4>/1-gpu --gemm_plugin auto --max_batch_size 1 --max_input_len 6144 --max_multimodal_len 6144 --max_seq_len 8192
```
In all, the above steps should generate both the vision engine and the LLM engine for Eagle2.5 model. 


## Eagle2.5 Inference with TensorRT-LLM and TensorRT  <a name="inference"></a>
Here, we also provide a sample inference script to run the Eagle2.5 with TensoRT engine and TensorRT-LLM engine all together, the following script outputs the generated text by given images and input prompt similar to the [inference script](../inference/inference.py) in pytorch:
```bash
python inference.py --model_path=<model_path> --vision_engine_path=<save_trt_engine_path> --llm_engine_path=<llm_engine_path>/x86/<fp16_or_int4>/1-gpu  --model_name=<model_name>
```