# DREAMPlace

Deep learning toolkit-enabled VLSI placement. 
With the analogy between nonlinear VLSI placement and deep learning training problem, this tool is developed with deep learning toolkit for flexibility and efficiency. 
The tool runs on both CPU and GPU. 
Over 30X speedup over the CPU implementation is achieved in global placement and legalization on ISPD 2005 contest benchmarks with a Nvidia Tesla V100 GPU. 

# Publications

* [Yibo Lin](http://yibolin.com), Shounak Dhar, [Wuxi Li](http://wuxili.net), Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**", 
  ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, Jun 2-6, 2019

# Dependency 

- Pytorch 0.4.1 or 1.0.0

- Python 2.7 or Python 3.5

- [Boost](www.boost.org)
    - Need to install and visible for linking

- [Limbo](https://github.com/limbo018/Limbo)
    - Integrated as a git submodule

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)
    - Integrated as a submodule

- [CUDA 9.1 or later](https://developer.nvidia.com/cuda-toolkit) (Optional)
    - If installed and found, GPU acceleration will be enabled. 
    - Otherwise, only CPU implementation is enabled. 

- [Cairo](https://github.com/freedesktop/cairo) (Optional)
    - If installed and found, the plotting functions will be faster by using C/C++ implementation. 
    - Otherwise, python implementation is used. 

- [NTUPlace3](http://eda.ee.ntu.edu.tw/research.htm) (Optional)
    - If the binary is provided, it can be used to perform detailed placement 

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

[CMake](https://cmake.org) is adopted as the makefile system. 
To build, go to the root directory. 
```
mkdir build 
cd build 
cmake ..
make 
make install
```

Third party submodules are automatically built except for [Boost](www.boost.org).

To clean, go to the root directory. 
```
rm -r build
```

Here are the available options for CMake. 
- CMAKE_INSTALL_PREFIX: installation directory
    - Example ```cmake -DCMAKE_INSTALL_PREFIX=path/to/your/directory```
- CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
    - Example ```cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60```

# How to Get Benchmarks

To get ISPD 2005 benchmarks, run the following script from the directory. 
```
python benchmarks/ispd2005.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded and the python dependency packages have been installed. 
Run with JSON configuration file for full placement in the root directory.  
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
