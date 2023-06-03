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

* [Yibo Lin](http://yibolin.com), Zixuan Jiang, [Jiaqi Gu](https://jeremiemelo.github.io), [Wuxi Li](http://wuxili.net), Shounak Dhar, Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**", 
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020 

* [Yibo Lin](http://yibolin.com), [Wuxi Li](http://wuxili.net), [Jiaqi Gu](https://jeremiemelo.github.io), Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**ABCDPlace: Accelerated Batch-based Concurrent Detailed Placement on Multi-threaded CPUs and GPUs**", 
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020 
  ([preprint](http://yibolin.com/publications/papers/ABCDPLACE_TCAD2020_Lin.pdf)) 

* [Yibo Lin](http://yibolin.com), [David Z. Pan](http://users.ece.utexas.edu/~dpan), Haoxing Ren and Brucek Khailany, 
  "**DREAMPlace 2.0: Open-Source GPU-Accelerated Global and Detailed Placement for Large-Scale VLSI Designs**", 
  China Semiconductor Technology International Conference (CSTIC), Shanghai, China, Jun, 2020  ([preprint](http://yibolin.com/publications/papers/PLACE_CSTIC2020_Lin.pdf))(Invited Paper)

* [Jiaqi Gu](https://jeremiemelo.github.io), Zixuan Jiang, [Yibo Lin](http://yibolin.com) and [David Z. Pan](http://users.ece.utexas.edu/~dpan), 
  "**DREAMPlace 3.0: Multi-Electrostatics Based Robust VLSI Placement with Region Constraints**", 
  IEEE/ACM International Conference on Computer-Aided Design (ICCAD), Nov 2-5, 2020
  ([preprint](http://yibolin.com/publications/papers/PLACE_ICCAD2020_Gu.pdf))

# Dependency 

- Python 3.5/3.6/3.7/3.8

- [Pytorch](https://pytorch.org/) 1.6/1.7/1.8
    - Other versions may also work, but not tested

- [GCC](https://gcc.gnu.org/)
    - Recommend GCC 5.1 or later. 
    - Other compilers may also work, but not tested. 

- [Boost](https://www.boost.org)
    - Need to install and visible for linking
  
- [Bison](https://www.gnu.org/software/bison) >= 3.3
    - Need to install

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
    - Code has been tested on GPUs with compute compatibility 6.0, 7.0, and 7.5. 
    - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices. 
    - The default compilation target is compatibility 6.0. This is the minimum requirement and lower compatibility is not supported for the GPU feature. 
    - For compatibility 7.0, it is necessary to set the CMAKE_CUDA_FLAGS to -gencode=arch=compute_70,code=sm_70. 
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

Run with GPU on Linux. 
```
docker run --gpus 1 -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```
Run with GPU on Windows.
```
docker run --gpus 1 -it -v /dreamplace limbo018/dreamplace:cuda bash
```
Run without GPU on Linux. 
```
docker run -it -v $(pwd):/DREAMPlace limbo018/dreamplace:cuda bash
```
Run without GPU on Windows.
```
docker run -it -v /dreamplace limbo018/dreamplace:cuda bash
```
6. ```cd /DREAMPlace```. 
7. Go to next section to complete building. 

## Build without Docker

[CMake](https://cmake.org) is adopted as the makefile system. 
To build, go to the root directory. 
```
mkdir build 
cd build 
cmake .. -DCMAKE_INSTALL_PREFIX=your_install_path -DPYTHON_EXECUTABLE=$(which python)
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

To get ISPD 2005 and 2015 benchmarks, run the following script from the directory. 
```
python benchmarks/ispd2005_2015.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded and the python dependency packages have been installed. 
Go to the **install directory** and run with JSON configuration file for full placement.  
```
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

Test individual `pytorch` op with the unit tests in the root directory. 
```
python unittest/ops/hpwl_unittest.py
```

# Configurations

Descriptions of options in JSON configuration file can be found by running the following command. 
```
python dreamplace/Placer.py --help
```

The list of options as follows will be shown. 

| JSON Parameter                   | Default                 | Description                                                                                                                                                       |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| aux_input                        | required for Bookshelf  | input .aux file                                                                                                                                                   |
| lef_input                        | required for LEF/DEF    | input LEF file                                                                                                                                                    |
| def_input                        | required for LEF/DEF    | input DEF file                                                                                                                                                    |
| verilog_input                    | optional for LEF/DEF    | input VERILOG file, provide circuit netlist information if it is not included in DEF file                                                                         |
| gpu                              | 1                       | enable gpu or not                                                                                                                                                 |
| num_bins_x                       | 512                     | number of bins in horizontal direction                                                                                                                            |
| num_bins_y                       | 512                     | number of bins in vertical direction                                                                                                                              |
| global_place_stages              | required                | global placement configurations of each stage, a dictionary of {"num_bins_x", "num_bins_y", "iteration", "learning_rate"}, learning_rate is relative to bin size  |
| target_density                   | 0.8                     | target density                                                                                                                                                    |
| density_weight                   | 1.0                     | initial weight of density cost                                                                                                                                    |
| gamma                            | 0.5                     | initial coefficient for log-sum-exp and weighted-average wirelength                                                                                               |
| random_seed                      | 1000                    | random seed                                                                                                                                                       |
| result_dir                       | results                 | result directory for output                                                                                                                                       |
| scale_factor                     | 0.0                     | scale factor to avoid numerical overflow; 0.0 means not set                                                                                                       |
| ignore_net_degree                | 100                     | ignore net degree larger than some value                                                                                                                          |
| gp_noise_ratio                   | 0.025                   | noise to initial positions for global placement                                                                                                                   |
| enable_fillers                   | 1                       | enable filler cells                                                                                                                                               |
| global_place_flag                | 1                       | whether use global placement                                                                                                                                      |
| legalize_flag                    | 1                       | whether use internal legalization                                                                                                                                 |
| detailed_place_flag              | 1                       | whether use internal detailed placement                                                                                                                           |
| stop_overflow                    | 0.1                     | stopping criteria, consider stop when the overflow reaches to a ratio                                                                                             |
| dtype                            | float32                 | data type, float32 | float64                                                                                                                                      |
| detailed_place_engine            |                         | external detailed placement engine to be called after placement                                                                                                   |
| detailed_place_command           | -nolegal -nodetail      | commands for external detailed placement engine                                                                                                                   |
| plot_flag                        | 0                       | whether plot solution or not                                                                                                                                      |
| RePlAce_ref_hpwl                 | 350000                  | reference HPWL used in RePlAce for updating density weight                                                                                                        |
| RePlAce_LOWER_PCOF               | 0.95                    | lower bound ratio used in RePlAce for updating density weight                                                                                                     |
| RePlAce_UPPER_PCOF               | 1.05                    | upper bound ratio used in RePlAce for updating density weight                                                                                                     |
| random_center_init_flag          | 1                       | whether perform random initialization around the center for global placement                                                                                      |
| sort_nets_by_degree              | 0                       | whether sort nets by degree or not                                                                                                                                |
| num_threads                      | 8                       | number of CPU threads                                                                                                                                             |
| dump_global_place_solution_flag  | 0                       | whether dump intermediate global placement solution as a compressed pickle object                                                                                 |
| dump_legalize_solution_flag      | 0                       | whether dump intermediate legalization solution as a compressed pickle object                                                                                     |

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
    - Support movable macros with Tetris-like macro legalization and min-cost flow refinement

* [2.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.1.0)
    - Support deterministic mode to ensure run-to-run determinism with minor runtime overhead

* [2.2.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.2.0)
    - Integrate routability optimization relying on NCTUgr from TCAD extension
    - Improved robustness on parallel CPU version 

* [3.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/3.0.0)
    - Support fence regions as published at ICCAD 2020
    - Add quadratic penalty to accelerate gradient descent at plateau during global placement
    - Inject noise to escape from saddle points during global placement
