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

Run with JSON configuration file for full placement 
```
python src/Placer.py test/simple.json
```

Test individual pytorch op
```
python src/pytorch/ops/density_potential/__init__.py
```

# Configurations

Descriptions of options in JSON configuration file are as follows. 
- opt_num_bins: one global optimization kernel with various settings, usually one is enough. 
- x in opt_num_bins: number of bins in x direction for the optimizer. 
- y in opt_num_bins: number of bins in y direction for the optimizer. 
- optimizer: kernel optimization algorithm for gradient descent. 
- target_density: target density for a design. 
- density_weight: initial density weight (lambda) in the objective (wirelength + lambda * density). 
- ignore_net_degree: only consider nets with degree smaller than a value in optimization. 
- legalize_flag: whether call internal legalization engine to legalize design or rely on NTUPlace.  
