FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
LABEL maintainer="Yibo Lin <yibolin@pku.edu.cn>"

# install system dependency 
RUN apt-get update \
        && apt-get install -y \
            wget \
            flex \
            bison \
            libcairo2-dev \
            libboost-all-dev 

# install cmake
ADD https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.sh /cmake-3.8.2-Linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.8.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version

# install python dependency 
RUN pip install \
        pyunpack>=0.1.2 \
        patool>=1.12 \
        matplotlib>=2.2.2 \
        cairocffi>=0.9.0 \
        pkgconfig>=1.4.0 \
        setuptools>=39.1.0 \
        scipy>=1.1.0 \
        numpy>=1.15.4 
