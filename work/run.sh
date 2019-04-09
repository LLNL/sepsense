#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} ./main.py \
 --param_path=./params/${2}.yaml
