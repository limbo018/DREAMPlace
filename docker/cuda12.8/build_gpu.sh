#!/usr/bin/env bash
# Compile DREAMPlace with CUDA inside the dreamplace:cuda12.8 image.
# Usage (from the repository root): docker/cuda12.8/build_gpu.sh .
# Env:   IMAGE (default dreamplace:cuda12.8), JOBS (default 4)
#
# The GPU must be visible here (--gpus all): CMake decides whether to build
# the CUDA kernels by calling torch.cuda.is_available() at configure time.
set -euo pipefail
REPO="$(realpath "$1")"
IMAGE="${IMAGE:-dreamplace:cuda12.8}"
JOBS="${JOBS:-4}"

# Pre-fetch the HeteroSTA archive with curl. CMake's own file(DOWNLOAD) inside
# the container was extremely slow (over 30 min for 23 MB, versus 2 s with curl),
# and thirdparty/HeteroSTA/CMakeLists.txt uses a local archive when one is present.
HS_CMAKE="$REPO/thirdparty/HeteroSTA/CMakeLists.txt"
HS_VER="$(sed -nE 's/^set\(HETEROSTA_VERSION "([^"]+)".*/\1/p' "$HS_CMAKE")"
HS_DATE="$(sed -nE 's/^set\(HETEROSTA_VERSION_DATE "([^"]+)".*/\1/p' "$HS_CMAKE")"
HS_ASSET="heterosta.v${HS_VER}.${HS_DATE}.tar.gz"
HS_DIR="$REPO/build/thirdparty/HeteroSTA"
mkdir -p "$HS_DIR"
if [ ! -s "$HS_DIR/$HS_ASSET" ]; then
  curl -fL -o "$HS_DIR/$HS_ASSET" \
    "https://github.com/HeteroSTA/releases/releases/download/v${HS_VER}-${HS_DATE}/${HS_ASSET}"
fi

docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -e JOBS="$JOBS" \
  -v "$REPO":/DREAMPlace -w /DREAMPlace "$IMAGE" bash -c '
set -euxo pipefail
python -c "
import torch
assert torch.cuda.is_available(), \"GPU not visible inside the container\"
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda,
      torch.cuda.get_device_name(0), \"capability\", torch.cuda.get_device_capability(0))
"
# Match the C++ ABI of the installed PyTorch (2.7 wheels use the CXX11 ABI);
# DREAMPlace defaults to CMAKE_CXX_ABI=0, which gives undefined c10 symbols at import.
# CUDA_CUDA_LIBRARY: link timing_heterosta against the driver stub shipped in the
# image (SONAME libcuda.so.1), so the real driver is loaded at run time and
# CMake does not depend on how the container toolkit exposes libcuda.so.
CXX_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/DREAMPlace/install -DPython_EXECUTABLE=$(which python) \
  -DCMAKE_CXX_ABI="$CXX_ABI" \
  -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so 2>&1 | tee cmake.log
grep -q "TORCH_ENABLE_CUDA=1" cmake.log || { echo "ERROR: CUDA was not enabled"; exit 1; }
grep -q "sm_120" cmake.log || { echo "ERROR: sm_120 flags were not added"; exit 1; }
make -j"$JOBS"
make install
'
