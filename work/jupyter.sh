#!/bin/bash

#jupyter notebook --ip 128.115.198.8 --allow-root
jupyter nbextension enable --py --sys-prefix widgetsnbextension
CUDA_VISIBLE_DEVICES=1 jupyter notebook --no-browser --port=8888 \
 --NotebookApp.token='' --allow-root --notebook-dir=./notebooks \
 --NotebookApp.iopub_data_rate_limit=10000000
