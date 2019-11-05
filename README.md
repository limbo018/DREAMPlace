# DREAMPlace

Deep learning toolkit-enabled VLSI placement. 
With the analogy between nonlinear VLSI placement and deep learning training problem, this tool is developed with deep learning toolkit for flexibility and efficiency. 
The tool runs on both CPU and GPU. 
Over ```30X``` speedup over the CPU implementation ([RePlAce](https://doi.org/10.1109/TCAD.2018.2859220)) is achieved in global placement and legalization on ISPD 2005 contest benchmarks with a Nvidia Tesla V100 GPU. 
DREAMPlace also integrates a GPU-accelerated detailed placer, *ABCDPlace*, which can achieve around ```16X``` speedup on million-size benchmarks over the widely-adopted sequential placer [NTUPlace3](https://doi.org/10.1109/TCAD.2008.923063) on CPU.

DREAMPlace runs on both CPU and GPU. If it is installed on a machine without GPU, only CPU support will be enabled with multi-threading. 

* Animation

| Bigblue4 | Density Map | Electric Potential | Electric Field |
| -------- | ----------- | ------------------ | -------------- |
| <img src=/images/bigblue4-nofiller_SLD.gif width=250> | ![Density Map](images/density_map_SLD.gif) | ![Electric Potential Map](images/potential_map_SLD.gif) | ![Electric Field Map](images/field_map_SLD.gif) |

* Reference Flow

<img src=/images/DREAMPlace2_flow.png width=600>

# Publications

* [Yibo Lin](http://yibolin.com), Shounak Dhar, [Wuxi Li](http://wuxili.net), Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**", 
  ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, Jun 2-6, 2019
  ([preprint](http://yibolin.com/publications/papers/PLACE_DAC2019_Lin.pdf)) ([slides](http://yibolin.com/publications/papers/PLACE_DAC2019_Lin.slides.pptx))

* [Yibo Lin](http://yibolin.com), Zixuan Jiang, Jiaqi Gu, [Wuxi Li](http://wuxili.net), Shounak Dhar, Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**", 
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020 (in submission)

* [Yibo Lin](http://yibolin.com), [Wuxi Li](http://wuxili.net), Jiaqi Gu, Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**ABCDPlace: Accelerated Batch-based Concurrent Detailed Placement on Multi-threaded CPUs and GPUs**", 
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020 (in submission)

# Dependency 

- Python 2.7 or Python 3.5/3.6/3.7

- [Pytorch](https://pytorch.org/) 1.0.0
    - Other version around 1.0.0 may also work, but not tested

- [GCC](https://gcc.gnu.org/)
    - Recommend GCC 5.1 or later. 
    - Other compilers may also work, but not tested. 

- [Boost](https://www.boost.org)
    - Need to install and visible for linking

- [Limbo](https://github.com/limbo018/Limbo)
    - Integrated as a git submodule

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)
    - Integrated as a submodule

- [CUB](https://github.com/NVlabs/cub)
    - Integrated as a git submodule

- [munkres-cpp](https://github.com/saebyn/munkres-cpp)
    - Integrated as a git submodule

- [CUDA 9.1 or later](https://developer.nvidia.com/cuda-toolkit) (Optional)
    - If installed and found, GPU acceleration will be enabled. 
    - Otherwise, only CPU implementation is enabled. 

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 6.0 and 7.0. 
    - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices. 
    - The default compilation target is compatibility 6.0. 
    For compatibility 7.0, it is necessary to set the CMAKE_CUDA_FLAGS to -gencode=arch=compute_70,code=sm_70. 

- [Cairo](https://github.com/freedesktop/cairo) (Optional)
    - If installed and found, the plotting functions will be faster by using C/C++ implementation. 
    - Otherwise, python implementation is used. 

- [NTUPlace3](http://eda.ee.ntu.edu.tw/research.htm) (Optional)
    - If the binary is provided, it can be used to perform detailed placement.

To pull git submodules in the root directory
```
git submodule init
git submodule update
```

Or alternatively, pull all the submodules when cloning the repository. 
```
git clone --recursive https://github.com/limbo018/DREAMPlace.git
```

# How to Install Python Dependency 

Go to the root directory. 
```
pip install -r requirements.txt 
```

# How to Build 

Two options are provided for building: with and without [Docker](https://hub.docker.com). 

## Build with Docker

You can use the Docker container to avoid building all the dependencies yourself. 
1. Install Docker on [Windows](https://docs.docker.com/docker-for-windows/), [Mac](https://docs.docker.com/docker-for-mac/) or [Linux](https://docs.docker.com/install/).
2. To enable the GPU features, install [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker); otherwise, skip this step.  
3. Navigate to the repository. 
4. Get the docker container with either of the following options. 
    - Option 1: pull from the cloud [limbo018/dreamplace](https://hub.docker.com/r/limbo018/dreamplace). 
    ```
    docker pull limbo018/dreamplace:cuda
    ```
    - Option 2: build the container. 
    ```
    docker build . --file Dockerfile --tag your_name/dreamplace:cuda
    ```
5. Enter bash environment of the container. Replace ```limbo018``` with your name if option 2 is chosen in the previous step. 

Run with GPU. 
```
docker run --gpus 1 -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```
Run without GPU. 
```
docker run -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```
6. ```cd /DREAMPlace```. 
7. Go to next section to complete building. 

## Build without Docker

[CMake](https://cmake.org) is adopted as the makefile system. 
To build, go to the root directory. 
```
mkdir build 
cd build 
cmake ..
make 
make install
```

Third party submodules are automatically built except for [Boost](https://www.boost.org).

To clean, go to the root directory. 
```
rm -r build
```

Here are the available options for CMake. 
- CMAKE_INSTALL_PREFIX: installation directory
    - Example ```cmake -DCMAKE_INSTALL_PREFIX=path/to/your/directory```
- CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
    - Example ```cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60```
- CMAKE_CXX_ABI: 0|1 for the value of _GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0. 
    - Example ```cmake -DCMAKE_CXX_ABI=0```
    - It must be consistent with the _GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch. 
    - PyTorch in default is compiled with _GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with _GLIBCXX_USE_CXX11_ABI=1. 

# How to Get Benchmarks

To get ISPD 2005 benchmarks, run the following script from the directory. 
```
python benchmarks/ispd2005.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded and the python dependency packages have been installed. 
Go to the **install directory** and run with JSON configuration file for full placement.  
```
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

Test individual pytorch op with the unitest in the root directory. 
```
python unitest/ops/hpwl_unitest.py
```

# Configurations

Descriptions of options in JSON configuration file can be found by running the following command. 
```
python dreamplace/Placer.py --help
```

# Authors

* [Yibo Lin](http://yibolin.com), supervised by [David Z. Pan](http://users.ece.utexas.edu/~dpan), composed the initial release. 
* [Zixuan Jiang](https://github.com/ZixuanJiang) and [Jiaqi Gu](https://github.com/JeremieMelo) improved the efficiency of the wirelength and density operators on GPU. 
* [Yibo Lin](http://yibolin.com) and [Jiaqi Gu](https://github.com/JeremieMelo) developed and integrated ABCDPlace for detailed placement. 
* **Pull requests to improve the tool are more than welcome.** We appreciate all kinds of contributions from the community. 

# Features

* [0.0.2](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.2)
    - Multi-threaded CPU and optional GPU acceleration support 

* [0.0.5](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.5)
    - Net weighting support through .wts files in Bookshelf format
    - Incremental placement support

* [0.0.6](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.6)
    - LEF/DEF support as input/output
    - Python binding and access to C++ placement database

* [1.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/1.0.0)
    - Improved efficiency for wirelength and density operators from TCAD extension

* [1.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/1.1.0)
    - Docker container for building environment

* [2.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.0.0)
    - Integrate ABCDPlace: multi-threaded CPU and GPU acceleration for detailed placement
    - Support independent set matching, local reordering, and global swap with run-to-run determinism on one machine
