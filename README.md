# DREAMPlace

Deep learning toolkit-enabled VLSI placement.
With the analogy between nonlinear VLSI placement and deep learning training problem, this tool is developed with deep learning toolkit for flexibility and efficiency.
The tool runs on both CPU and GPU.
Over `30X` speedup over the CPU implementation ([RePlAce](https://doi.org/10.1109/TCAD.2018.2859220)) is achieved in global placement and legalization on ISPD 2005 contest benchmarks with a Nvidia Tesla V100 GPU.
DREAMPlace also integrates a GPU-accelerated detailed placer, _ABCDPlace_, which can achieve around `16X` speedup on million-size benchmarks over the widely-adopted sequential placer [NTUPlace3](https://doi.org/10.1109/TCAD.2008.923063) on CPU.

- [DREAMPlace](#dreamplace)
- [Publications](#publications)
- [Dependency](#dependency)
- [How to Install Python Dependency](#how-to-install-python-dependency)
- [How to Build](#how-to-build)
  - [Build with Docker](#build-with-docker)
  - [Build without Docker](#build-without-docker)
- [How to Get Benchmarks](#how-to-get-benchmarks)
- [How to Run](#how-to-run)
- [Configurations](#configurations)
- [Authors](#authors)
- [Features](#features)

DREAMPlace runs on both CPU and GPU. If it is installed on a machine without GPU, only CPU support will be enabled with multi-threading.

- Animation

| Bigblue4                                              | Density Map                                | Electric Potential                                      | Electric Field                                  |
| ----------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- | ----------------------------------------------- |
| <img src=/images/bigblue4-nofiller_SLD.gif width=250> | ![Density Map](images/density_map_SLD.gif) | ![Electric Potential Map](images/potential_map_SLD.gif) | ![Electric Field Map](images/field_map_SLD.gif) |

- Reference Flow

<img src=images/DREAMPlace4.1_flow.png width=600>

# Publications

- [Yibo Lin](http://yibolin.com), Shounak Dhar, [Wuxi Li](http://wuxili.net), Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan),
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**",
  ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, Jun 2-6, 2019
  ([preprint](http://yibolin.com/publications/papers/PLACE_DAC2019_Lin.pdf)) ([slides](http://yibolin.com/publications/papers/PLACE_DAC2019_Lin.slides.pptx))

- [Yibo Lin](http://yibolin.com), Zixuan Jiang, [Jiaqi Gu](https://jeremiemelo.github.io), [Wuxi Li](http://wuxili.net), Shounak Dhar, Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan),
  "**DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement**",
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020

- [Yibo Lin](http://yibolin.com), [Wuxi Li](http://wuxili.net), [Jiaqi Gu](https://jeremiemelo.github.io), Haoxing Ren, Brucek Khailany and [David Z. Pan](http://users.ece.utexas.edu/~dpan),
  "**ABCDPlace: Accelerated Batch-based Concurrent Detailed Placement on Multi-threaded CPUs and GPUs**",
  IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020
  ([preprint](http://yibolin.com/publications/papers/ABCDPLACE_TCAD2020_Lin.pdf))

- [Yibo Lin](http://yibolin.com), [David Z. Pan](http://users.ece.utexas.edu/~dpan), Haoxing Ren and Brucek Khailany,
  "**DREAMPlace 2.0: Open-Source GPU-Accelerated Global and Detailed Placement for Large-Scale VLSI Designs**",
  China Semiconductor Technology International Conference (CSTIC), Shanghai, China, Jun, 2020 ([preprint](http://yibolin.com/publications/papers/PLACE_CSTIC2020_Lin.pdf))(Invited Paper)

- [Jiaqi Gu](https://jeremiemelo.github.io), Zixuan Jiang, [Yibo Lin](http://yibolin.com) and [David Z. Pan](http://users.ece.utexas.edu/~dpan),
  "**DREAMPlace 3.0: Multi-Electrostatics Based Robust VLSI Placement with Region Constraints**",
  IEEE/ACM International Conference on Computer-Aided Design (ICCAD), Nov 2-5, 2020
  ([preprint](http://yibolin.com/publications/papers/PLACE_ICCAD2020_Gu.pdf))

- [Peiyu Liao](https://enzoleo.github.io), [Siting Liu](https://lusica1031.github.io), Zhitang Chen, Wenlong Lv, [Yibo Lin](http://yibolin.com) and [Bei Yu](https://www.cse.cuhk.edu.hk/~byu/),
  "**DREAMPlace 4.0: Timing-driven Global Placement with Momentum-based Net Weighting**",
  IEEE/ACM Proceedings Design, Automation and Test in Eurpoe (DATE), Antwerp, Belgium, Mar 14-23, 2022
  ([preprint](https://yibolin.com/publications/papers/PLACE_DATE2022_Liao.pdf))

- Yifan Chen, [Zaiwen Wen](http://faculty.bicmr.pku.edu.cn/~wenzw/), [Yun Liang](https://ericlyun.github.io/), [Yibo Lin](http://yibolin.com),
  "**Stronger Mixed-Size Placement Backbone Considering Second-Order Information**",
  IEEE/ACM International Conference on Computer-Aided Design (ICCAD), San Francisco, CA, Oct, 2023
  ([preprint](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf))

# Dependency

- [Python](https://www.python.org/) 3.5/3.6/3.7/3.8/3.9

- [Pytorch](https://pytorch.org/) 1.6/1.7/1.8/2.0

  - Other versions may also work, but not tested

- [GCC](https://gcc.gnu.org/)

  - Recommend GCC 7.5 (with `c++17` support).
  - Do not recommend GCC 9 or later due to backward compatibility issues.
  - Other compilers may also work, but not tested.

- [Boost](https://www.boost.org) >= 1.55.0
  - Need to install and visible for linking
- [Bison](https://www.gnu.org/software/bison) >= 3.3

  - Need to install

- [Limbo](https://github.com/limbo018/Limbo)

  - Integrated as a git submodule

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)

  - Integrated as a submodule

- [OpenTimer](https://github.com/OpenTimer/OpenTimer)

  - [Modified version](https://github.com/enzoleo/OpenTimer) for timing optimization
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
5. Enter bash environment of the container. Replace `limbo018` with your name if option 2 is chosen in the previous step.

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

6. `cd /DREAMPlace`.
7. Go to [next section](#build-without-docker) to complete building within the container.

## Build without Docker

[CMake](https://cmake.org) is adopted as the makefile system.
To build, go to the root directory.

```
mkdir build
cd build # we call this <build directory>
cmake .. -DCMAKE_INSTALL_PREFIX=<installation directory> -DPython_EXECUTABLE=$(which python)
make
make install
```
Where `<build directory>` is the directory where you compile the code, and `<installation directory>` is the directory where you want to install DREAMPlace (e.g., `../install`).
Third party submodules are automatically built except for [Boost](https://www.boost.org).

To clean, go to the root directory.

```
rm -r build
```
`<build directory>` can be removed after installation if you do not need incremental compilation later. 

Here are the available options for CMake.

- CMAKE_INSTALL_PREFIX: installation directory
  - Example `cmake -DCMAKE_INSTALL_PREFIX=path/to/your/directory`
- CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
  - Example `cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60`
- CMAKE_CXX_ABI: 0|1 for the value of \_GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0.
  - Example `cmake -DCMAKE_CXX_ABI=0`
  - It must be consistent with the \_GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch.
  - PyTorch in default is compiled with \_GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with \_GLIBCXX_USE_CXX11_ABI=1.

# How to Get Benchmarks

To get ISPD 2005 and 2015 benchmarks, run the following script from the directory.

```
python benchmarks/ispd2005_2015.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded and the python dependency packages have been installed.
Go to the **install directory** and run with JSON configuration file for full placement.

```
cd <installation directory>
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

Test individual `pytorch` op with the unit tests in the root directory.

```
cd <installation directory>
python unittest/ops/hpwl_unittest.py
```

# Configurations

Descriptions of options in JSON configuration file can be found by running the following command.

```
cd <installation directory>
python dreamplace/Placer.py --help
```

The list of options as follows will be shown.

| JSON Parameter | Default                | Description                                                                               |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------- |
| aux_input      | required for Bookshelf | input .aux file                                                                           |
| lef_input      | required for LEF/DEF   | input LEF file                                                                            |
| def_input      | required for LEF/DEF   | input DEF file                                                                            |
| verilog_input  | optional for LEF/DEF   | input VERILOG file, provide circuit netlist information if it is not included in DEF file |
| gpu            | 1                      | enable gpu or not                                                                         |

...

# Authors

- [Yibo Lin](http://yibolin.com), supervised by [David Z. Pan](http://users.ece.utexas.edu/~dpan), composed the initial release.
- [Zixuan Jiang](https://github.com/ZixuanJiang) and [Jiaqi Gu](https://github.com/JeremieMelo) improved the efficiency of the wirelength and density operators on GPU.
- [Yibo Lin](http://yibolin.com) and [Jiaqi Gu](https://github.com/JeremieMelo) developed and integrated ABCDPlace for detailed placement.
- [Peiyu Liao](https://enzoleo.github.io) and [Siting Liu](https://lusica1031.github.io) developed and integrated timing optimization in global placement for DREAMPlace 4.0.
- Yifan Chen developed the 2-stage flow and improved the optimizer for macro placement in DREAMPlace 4.1. Set ```use_bb``` to 1 to turn on BB-step and ```macro_place_flag``` to 1 to enable 2-stage flow for macro placement. 
- Yiting Liu contributed the GiFt operator for placement initialization, published at ICCAD 2024. Set ```gift_init_flag``` to 1 to turn on this feature, and use ```gift_init_scale``` to control the scale parameter of this operator. 
```
Yiting Liu, Hai Zhou, Jia Wang, Fan Yang, Xuan Zeng, Li Shang, 
"The Power of Graph Signal Processing for Chip Placement Acceleration", 
  IEEE/ACM International Conference on Computer-Aided Design (ICCAD), New Jersey, USA, Oct, 2024
(Thanks for contributing the source code!)
```

- **Pull requests to improve the tool are more than welcome.** We appreciate all kinds of contributions from the community.

# Features

- [0.0.2](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.2)

  - Multi-threaded CPU and optional GPU acceleration support

- [0.0.5](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.5)

  - Net weighting support through .wts files in Bookshelf format
  - Incremental placement support

- [0.0.6](https://github.com/limbo018/DREAMPlace/releases/tag/0.0.6)

  - LEF/DEF support as input/output
  - Python binding and access to C++ placement database

- [1.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/1.0.0)

  - Improved efficiency for wirelength and density operators from TCAD extension

- [1.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/1.1.0)

  - Docker container for building environment

- [2.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.0.0)

  - Integrate ABCDPlace: multi-threaded CPU and GPU acceleration for detailed placement
  - Support independent set matching, local reordering, and global swap with run-to-run determinism on one machine
  - Support movable macros with Tetris-like macro legalization and min-cost flow refinement

- [2.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.1.0)

  - Support deterministic mode to ensure run-to-run determinism with minor runtime overhead

- [2.2.0](https://github.com/limbo018/DREAMPlace/releases/tag/2.2.0)

  - Integrate routability optimization relying on NCTUgr from TCAD extension
  - Improved robustness on parallel CPU version

- [3.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/3.0.0)

  - Support fence regions as published at ICCAD 2020
  - Add quadratic penalty to accelerate gradient descent at plateau during global placement
  - Inject noise to escape from saddle points during global placement

- [4.0.0](https://github.com/limbo018/DREAMPlace/releases/tag/4.0.0)
  - Support timing optimization in global placement as published at DATE 2022
  - Add momentum-based net weighting strategy
  - Integrate OpenTimer for static timing analysis
  - Tested under ICCAD 2015 contest benchmarks (see test/iccad2015.ot)

- [4.1.0](https://github.com/limbo018/DREAMPlace/releases/tag/4.1.0)
  - Support BB step and 2-stage macro placement flow as published at ICCAD 2023 (Need to set ```use_bb``` to 1 to turn on BB-step and ```macro_place_flag``` to 1 to enable 2-stage flow for macro placement)
  - Tested under ISPD 2005 with all fixed macros and IO pads regarded as movable macros (see test/ispd2005free) and MMS benchmarks (see test/mms)
 
- [4.2.0](https://github.com/limbo018/DREAMPlace/releases/tag/4.2.0)
  - Support GiFt initialization as published at ICCAD 2024
 
# Reference Results for Macro Placement

Recently, many studies chose DREAMPLace for macro placement, e.g., [[Cheng+, NeurIPS2021](https://arxiv.org/abs/2111.00234)], [[Lai+, NeurIPS2023](https://arxiv.org/pdf/2211.13382)], etc. However, the results reported on the same benchmarks vary significantly from one work to another. For better comparison, we provide the results collected from our GPU machine for reference. If your results deviate significantly (i.e., >5% longer HPWL) from the following numbers, something may be wrong. We recommend you to contact us with logs for validation.

Note that DREAMPlace 4.1.0 only implements the BB step and 2-stage flow proposed in [[Chen+, ICCAD2023](https://ieeexplore.ieee.org/document/10323700)]. 

[ISPD2005 benchmark](https://dl.acm.org/doi/10.1145/1629911.1630028) with all fixed macros and IO pads regarded as movable macros. It can be downloaded from [here](https://www.dropbox.com/scl/fi/01jvzui9hv0aa4krnd8lm/ispd2005free.zip?rlkey=ijwspusl9onncnsu5j4na4tqe&st=l44f3dnw&dl=0).
<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">DREAMPlace 4.0</th>
    <th colspan="3">DREAMPlace 4.1.0</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Iterations</td>
    <td>HPWL(x10^6)</td>
    <td>Time(s)</td>
    <td>Iterations</td>
    <td>HPWL(x10^6)</td>
    <td>Time(s)</td>
  </tr>
  <tr>
    <td>adaptec1</td>
    <td>600</td>
    <td>101.3</td>
    <td>26.3</td>
    <td>748</td>
    <td>68.2 </td>
    <td>27.6</td>
  </tr>
  <tr>
    <td>adaptec2</td>
    <td>588*</td>
    <td>137.5*</td>
    <td>40.6*</td>
    <td>784</td>
    <td>86.3 </td>
    <td>40.1</td>
  </tr>
  <tr>
    <td>adaptec3</td>
    <td>765</td>
    <td>179.5</td>
    <td>54.1</td>
    <td>894</td>
    <td>144.0 </td>
    <td>56.1</td>
  </tr>
  <tr>
    <td>adaptec4</td>
    <td>876</td>
    <td>153.3</td>
    <td>48.9</td>
    <td>872</td>
    <td>140.8 </td>
    <td>57.3</td>
  </tr>
  <tr>
    <td>bigblue1</td>
    <td>699</td>
    <td>86.2</td>
    <td>23.5</td>
    <td>813</td>
    <td>82.0 </td>
    <td>25.5</td>
  </tr>
  <tr>
    <td>bigblue2</td>
    <td>1267*</td>
    <td>2426.7*</td>
    <td>679.4*</td>
    <td>869</td>
    <td>98.1 </td>
    <td>193.4</td>
  </tr>
  <tr>
    <td>bigblue3</td>
    <td>1207</td>
    <td>330.2</td>
    <td>115.4</td>
    <td>1307</td>
    <td>288.8 </td>
    <td>140.1</td>
  </tr>
  <tr>
    <td>bigblue4</td>
    <td>1581</td>
    <td>820.1</td>
    <td>239.6</td>
    <td>1875</td>
    <td>610.0 </td>
    <td>234.5</td>
  </tr>
  <tr>
    <td>average ratio</td>
    <td>0.937</td>
    <td>4.211</td>
    <td>1.258</td>
    <td>1.000</td>
    <td>1.000</td>
    <td>1.000</td>
  </tr>
</tbody>
</table>

[MMS benchmark](https://dl.acm.org/doi/10.1145/1629911.1630028) (modified from ISPD2005 benchmarks with movable macros and fixed IO pads)

Our modified version can be downloaded from [here](https://www.dropbox.com/scl/fi/23i1jlmc9xhvfnj0jv82k/mms_mod.zip?rlkey=x4c0a4r877ii5ychu5lk4ur48&dl=0).

<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">DREAMPlace 4.0</th>
    <th colspan="3">DREAMPlace 4.1.0</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Iterations</td>
    <td>HPWL(x10^6)</td>
    <td>Time(s)</td>
    <td>Iterations</td>
    <td>HPWL(x10^6)</td>
    <td>Time(s)</td>
  </tr>
  <tr>
    <td>adaptec1</td>
    <td>607</td>
    <td>65.3 </td>
    <td>17.8 </td>
    <td>746</td>
    <td>64.7 </td>
    <td>25.8</td>
  </tr>
  <tr>
    <td>adaptec2</td>
    <td>569</td>
    <td>79.3 </td>
    <td>28.5 </td>
    <td>734</td>
    <td>75.8 </td>
    <td>35.8</td>
  </tr>
  <tr>
    <td>adaptec3</td>
    <td>659</td>
    <td>158.1 </td>
    <td>44.6 </td>
    <td>755</td>
    <td>153.3 </td>
    <td>38.9</td>
  </tr>
  <tr>
    <td>adaptec4</td>
    <td>735</td>
    <td>141.7 </td>
    <td>46.8 </td>
    <td>782</td>
    <td>142.4 </td>
    <td>47.5</td>
  </tr>
  <tr>
    <td>adaptec5</td>
    <td>1053</td>
    <td>326.3</td>
    <td>63.8</td>
    <td>1405</td>
    <td>337.6 </td>
    <td>78.4</td>
  </tr>
  <tr>
    <td>bigblue1</td>
    <td>646</td>
    <td>85.4 </td>
    <td>21.3 </td>
    <td>809</td>
    <td>85.3 </td>
    <td>28.9</td>
  </tr>
  <tr>
    <td>bigblue2</td>
    <td>638</td>
    <td>125.3 </td>
    <td>42.0 </td>
    <td>773</td>
    <td>125.4 </td>
    <td>48.4</td>
  </tr>
  <tr>
    <td>bigblue3</td>
    <td>911</td>
    <td>279.3 </td>
    <td>112.5 </td>
    <td>1097</td>
    <td>273.8 </td>
    <td>136.1</td>
  </tr>
  <tr>
    <td>bigblue4</td>
    <td>1189</td>
    <td>648.8 </td>
    <td>172.4 </td>
    <td>1515</td>
    <td>643.2 </td>
    <td>215.4</td>
  </tr>
  <tr>
    <td>newblue1</td>
    <td>574</td>
    <td>62.8 </td>
    <td>22.5 </td>
    <td>749</td>
    <td>62.0 </td>
    <td>30.4</td>
  </tr>
  <tr>
    <td>newblue2</td>
    <td>730</td>
    <td>155.5 </td>
    <td>34.8 </td>
    <td>861</td>
    <td>156.1 </td>
    <td>43.9</td>
  </tr>
  <tr>
    <td>newblue3</td>
    <td>1318*</td>
    <td>597.3*</td>
    <td>55.71*</td>
    <td>830</td>
    <td>270.6 </td>
    <td>72.8</td>
  </tr>
  <tr>
    <td>newblue4</td>
    <td>1009</td>
    <td>246.2</td>
    <td>52.6</td>
    <td>1274</td>
    <td>245.8 </td>
    <td>53.9</td>
  </tr>
  <tr>
    <td>newblue5</td>
    <td>1254</td>
    <td>444.2 </td>
    <td>99.4 </td>
    <td>1537</td>
    <td>446.4 </td>
    <td>134.9</td>
  </tr>
  <tr>
    <td>newblue6</td>
    <td>929</td>
    <td>410.6 </td>
    <td>96.1 </td>
    <td>1157</td>
    <td>409.3 </td>
    <td>115.1</td>
  </tr>
  <tr>
    <td>newblue7</td>
    <td>1077</td>
    <td>903.6 </td>
    <td>184.1 </td>
    <td>1578</td>
    <td>903.2 </td>
    <td>235.1</td>
  </tr>
  <tr>
    <td>average ratio</td>
    <td>0.855</td>
    <td>1.081</td>
    <td>0.830</td>
    <td>1.000</td>
    <td>1.000</td>
    <td>1.000</td>
  </tr>
</tbody>
</table>
`*` denotes divergence or legalization failure. 
Note that if you observe divergence or legalization errors in the log, then the results may not be representative. 

