# DREAMPlace
Deep learning toolkit-enabled VLSI placement

# Dependency 

- CUDA 9.1 or later

- Pytorch 0.4.1

- Python 2.7

- [Boost](www.boost.org)
    - Need to install and visible for linking

- [Limbo](https://github.com/limbo018/Limbo)
    - Integrated as a git submodule

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)
    - Integrated as a submodule

To pull git submodules in the root directory
```
git submodule init
git submodule update
```

# How to Build 

## User Mode 

[CMake](https://cmake.org) is adopted as the makefile system for end-uers. 
To build, go to the root directory. 
```
mkdir build 
cd build 
cmake ..
make 
```

Third party submodules are automatically built except for [Boost](www.boost.org).

To clean, go to the root directory. 
```
cd build 
make clean
```
Please note that simply removing the build folder will not completely clean the environment, because the python submodules have been installed to the python environment and need to be uninstalled. 

## Developer Mode 

Developers who prefer to have detailed control over the building process of each ops may want to use the handwritten makefile system. 
It supports incremental building of each ops for development. 
To build, run make in the root directory. 
```
make 
```
GCC 4.8 or later is preferred. 
Export CC and CXX environment variables for custom gcc and g++ path, respectively. 

Third party submodules are automatically built except for [Boost](www.boost.org).

To clean, run make clean in the root directory. 
```
make clean
```

# How to Get Benchmarks

To get ISPD 2005 benchmarks, run the following script from the directory. 
```
python benchmarks/ispd2005.py
```

# How to Run

Before running, make sure the benchmarks have been downloaded. 
Run with JSON configuration file for full placement.  
```
python dreamplace/Placer.py test/ispd2005/adaptec1.json
```

Test individual pytorch op. 
```
python dreamplace/ops/density_potential/__init__.py
```

# Configurations

Descriptions of options in JSON configuration file can be found by running the following command. 
```
python dreamplace/Placer.py 
```
