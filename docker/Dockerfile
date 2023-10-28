FROM ubuntu:22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt update && apt upgrade -y &&\
    apt install -y \
    git \
    wget \
    unzip \
    libpython3-dev \
    python3-pip \ 
    cmake \
    python3-dev \
    python3-numpy \
    python3-matplotlib \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://ceres-solver.googlesource.com/ceres-solver \
    &&  cd ceres-solver \
    &&  mkdir release \
    &&  cd release \
    &&  cmake .. \
    &&  make -j3 \
    &&  make install