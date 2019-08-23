# Start from NVIDIA?CUDA image with CUDA 10.0, Ubuntu 18.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Need to clear these sources for apt to work properly (necessary for nvidia/cuda base images)
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

# Install ubuntu packages
RUN apt-get update && apt-get install -y \
    git \
    python3.6 \
    python3.6-dev \
    build-essential \
    cmake \
    wget \
    pkg-config \
    man-db \
    vim \
    eog \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Install pip for python3.6
RUN apt-get install -y python3.6-distutils
WORKDIR /opt
RUN wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python3.6 get-pip.py

## Install python libs
RUN pip3 install ipython
RUN pip3 install jupyter
# Jupyter extensions
RUN pip3 install jupyter_contrib_nbextensions && jupyter contrib nbextension install
# Data processing libs
RUN pip3 install numpy==1.15.4
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install scikit-learn
RUN pip3 install opencv-python
# Libs for progressbar and logging
RUN pip3 install dill
RUN pip3 install progressbar2

# Install pytorch and torchvision
WORKDIR /opt
RUN wget --no-check-certificate http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision

# Add tkinter back in
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.6-tk

# Install tabulate
RUN pip3 install tabulate

# Install stats
RUN pip3 install statsmodels

# Install tifffile
RUN pip3 install tifffile

# Install libsvm
RUN apt-get install -y libsvm-dev

# Install BRISQUE
WORKDIR /opt
RUN git clone https://github.com/spmallick/learnopencv.git
WORKDIR /opt/learnopencv/ImageMetrics
RUN git reset --hard 799808f7301485714335026daca890251d090d7c
RUN cd Python/libsvm && make && cd python && make

# Extra
RUN apt-get install -y python3-gdal

# libSM needed for OpenCV
RUN apt-get install -y libsm-dev

# Create user with your username
RUN useradd --create-home --shell /bin/bash --groups sudo username
RUN echo "username:password" | chpasswd
