#!/bin/bash

docker run --runtime=nvidia \
 -it --rm \
 --name sepsense_`date +%F_%H-%M-%S` \
 --net=host \
 --ipc=host \
 -v /dev/shm:/dev/shm \
 -v ${HOME}/.vimrc:/root/.vimrc \
 -v ${HOME}/.vim:/root/.vim \
 -v `pwd`/work:/home/username/work \
 -v $(pwd)/data:/data/sepsense \
 -w /home/username/work \
 --env="DISPLAY" \
 --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
 --ulimit core=0 \
 --privileged \
 sepsense:v0.8 /bin/bash
