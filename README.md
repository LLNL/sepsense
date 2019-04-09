# Remote Sensor Design for Visual Recognition <br/> with Convolutional Neural Networks
TODO: edit down, include same version in eSoftware?
While deep learning technologies for computer vision have developed rapidly since 2012, modeling of remote sensing systems has remained focused around human vision. In particular, remote sensing systems are usually constructed to optimize sensing cost-quality trade-offs with respect to human image interpretability. While some recent studies have explored remote sensing system design as a function of simple computer vision algorithm performance, there has been little work relating this design to the state-of-the-art in computer vision: deep learning with convolutional neural networks. We develop experimental systems to conduct this analysis, showing results with modern deep learning algorithms and recent overhead image data. Our results are compared to standard image quality measurements based on human visual perception, and we conclude not only that machine and human interpretability differ significantly, but that computer vision performance is largely self-consistent across a range of disparate conditions.

This repository contains all the code required to replicate the results of our research paper, and the corresponding docker environment in which that code can be executed. All code is written in Python3 and utilizes the PyTorch library (v1.0) for neural network model training and evaluation. Jupyter notebooks are used to visualize images transformed with our code, in addition to plotting experimental results. We also include the specific parameter files used to execute our experiments, with the intent that these experiments can be replicated.

## 

## Instructions
### Setup Environment
The code in this repository is meant to run inside a docker environment. A Dockerfile is included in the ROOT/docker directory which contains an environment sufficient to run all included code. This Dockerfile is based on the official nvidia/cuda image with Ubuntu 18.04 and CUDA 10.

To build the docker image:
```
cd docker && ./build.sh
```
### Download Data
This code is centered around the [Functional Map of the World (fMoW) Dataset](https://github.com/fMoW/dataset). The pre-processing and partitioning scripts assume that this dataset is being used, but most of the code is general, and can be adapted to other datasets.
Instructions for downloading the fMoW dataset are included in the [corresponding repo](https://github.com/fMoW/dataset).
### Pre-Process Data
### Run Experiment
### View Results

## License
TBD
