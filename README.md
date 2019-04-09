# Remote Sensor Design for Visual Recognition with Convolutional Neural Networks
TODO: edit down, include same version in eSoftware?
While deep learning technologies for computer vision have developed rapidly since 2012, modeling of remote sensing systems has remained focused around human vision. In particular, remote sensing systems are usually constructed to optimize sensing cost-quality trade-offs with respect to human image interpretability. While some recent studies have explored remote sensing system design as a function of simple computer vision algorithm performance, there has been little work relating this design to the state-of-the-art in computer vision: deep learning with convolutional neural networks. We develop experimental systems to conduct this analysis, showing results with modern deep learning algorithms and recent overhead image data. Our results are compared to standard image quality measurements based on human visual perception, and we conclude not only that machine and human interpretability differ significantly, but that computer vision performance is largely self-consistent across a range of disparate conditions.

This repository contains all the code required to replicate the results of our research paper, and the corresponding docker environment in which that code can be executed. All code is written in Python3 and utilizes the PyTorch library (v1.0) for neural network model training and evaluation. Jupyter notebooks are used to visualize images transformed with our code, in addition to plotting experimental results. We also include the specific parameter files used to execute our experiments, with the intent that these experiments can be replicated.

## Capabilities
This code has the ability to reproduce the experiments from our research paper, and many of our visualizations.

### Conduct Experiments
The core purpose of the code is to measure how different sensor design configurations will affect recognition performance on the imagery these sensors will produce. For example, how will the imagery of a satellite sensor in orbit at 100km compare to that of one with an orbit at 500km? We mainly observe the effect of two optical parameters: focal length and aperture diameter. However, the system implemented in this code can explore many different factors which will affect image quality and recognition performance:
- Modify sensor parameters
- Modify image pre-processing methods
- Use different CNN architectures
- Solve different recognition problems
    - Classification
    - Retrieval
- Use different machine learning objectives
    - Cross-Entropy Loss
    - Triplet Margin Loss

### Analyze Results

#### Show Transformations

#### Plot Results

#### Visualize Retrieval

## Run Instructions
This section contains the instructions required to replicate our research results, and conduct new experiments.

### Setup Environment
The code in this repository is meant to run inside a docker environment. A Dockerfile is included in the REPO_ROOT/docker directory which contains an environment sufficient to run all included code. This Dockerfile is based on the official nvidia/cuda image with Ubuntu 18.04 and CUDA 10.

To build the docker image:
```
host:REPO_ROOT$ cd docker && ./build.sh
```
### Download Data
This code is centered around the [Functional Map of the World (fMoW) Dataset](https://github.com/fMoW/dataset). The pre-processing and partitioning scripts assume that this dataset is being used, but most of the code is general, and can be adapted to other datasets.

Instructions for downloading the fMoW dataset are included in the [corresponding repo](https://github.com/fMoW/dataset). You will need to download the train and val partitions of the "fMoW-full" dataset. This data contains the 4 and 8-band multispectral tiff files needed in this code.

After downloading the data, the volume mounts in your docker container run script should be edited to put the dataset in a specific location:
```
 -v /PATH/TO/FMOW/DATA:/data/fmow/full \
```

### Run Docker Container
To execute the included Python scripts, start a docker container based on the built image. The REPO_ROOT/dev.sh script provides a template for a docker run script. You may also use [Kubernetes](https://kubernetes.io/), or another container orchestration system.

### Pre-Process Data
To create a partition of the data usable by this code, execute the following within the docker container:
```
container:/home/username/work# ./prep.py
```
This partition will be saved to REPO_ROOT/work/partition/ms/train_items_35_1x10-fold.json by default.

### Run Experiment

#### Configure Parameters

#### Start Trial

### View Results

## License
TBD
