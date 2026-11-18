# DREAMPlace build environment for NVIDIA Blackwell GPUs (RTX 50xx, sm_120).
#
# This image only contains the toolchain and the Python packages.
# DREAMPlace itself is compiled afterwards with `docker run --gpus all`
# (see build_gpu.sh): cmake/TorchExtension.cmake enables CUDA only when
# torch.cuda.is_available() is true, and GPUs are not visible during `docker build`.
#
# Build (from this directory):  docker build -t dreamplace:cuda12.8 .
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies listed in the DREAMPlace README.
# Ubuntu 22.04 ships Bison 3.8 (README needs >= 3.3) and GCC 11 (supported by CUDA 12.8).
# Boost: thirdparty/Limbo needs the graph and regex components.
# tcl: thirdparty/OpenTimer requires tclsh at configure time.
# libfl-dev: provides FlexLexer.h; it is only a "Recommends" of flex, so
# --no-install-recommends would otherwise skip it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        flex \
        libfl-dev \
        bison \
        libboost-graph-dev \
        libboost-regex-dev \
        tcl \
        zlib1g-dev \
        libcairo2-dev \
        python3 \
        python3-dev \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.26.4 is the version the sm_120 support (commit dbec4cd) was written against,
# and it still provides FindCUDA, which the DREAMPlace build uses.
RUN pip install cmake==3.26.4

# PyTorch 2.7 is the first release with CUDA 12.8 wheels that support Blackwell.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Python dependencies from requirements.txt (torch excluded, installed above).
# numpy < 2 keeps the older code paths in DREAMPlace safe.
RUN pip install \
        "pyunpack>=0.1.2" \
        "patool>=1.12" \
        "matplotlib>=2.2.2" \
        "cairocffi>=0.9.0" \
        "pkgconfig>=1.4.0" \
        "setuptools>=39.1.0" \
        "scipy>=1.1.0" \
        "numpy>=1.15.4,<2" \
        "shapely>=1.7.0" \
        "torch_optimizer==0.3.0" \
        "ncg_optimizer==0.2.2"

WORKDIR /DREAMPlace
