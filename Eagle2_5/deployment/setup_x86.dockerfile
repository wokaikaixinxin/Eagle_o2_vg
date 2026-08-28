FROM nvcr.io/nvidia/tensorrt:25.06-py3

RUN chmod 1777 /tmp
# RUN apt update && apt install -y libturbojpeg libsm6 libxext6 -y
# RUN apt install libgl1-mesa-glx libsm6 libxext6  -y

# RUN pip install setuptools==65.5.1 debugpy einops tqdm numpy pandas
# RUN pip install llvmlite==0.41.0
# RUN pip install numba==0.58.0 scikit-image==0.18.3 "matplotlib<3.6.0"

RUN pip install onnx onnxsim onnxruntime onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install pycuda fvcore timm peft liger_kernel
RUN pip install transformers==4.51.0 accelerate
RUN pip install qwen-vl-utils[decord]==0.0.8
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=16 pip install flash-attn
RUN pip install dill==0.3.7
RUN pip install mpi4py

# Update cmake version from 3.24 to 3.27
RUN apt update; \
    apt install -y build-essential libssl-dev; \
    wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6.tar.gz; \
    tar xf cmake-3.27.6.tar.gz; \
    cd cmake-3.27.6; \
    ./bootstrap; \
    make -j$(nproc); \
    make install

RUN apt install ninja-build
