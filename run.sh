#!/usr/local/bin/bash

set -e

declare -A datasets=(
    ["transl-domain2"]="https://www.dropbox.com/s/kp48kccrv9taaom/icw-translation-left2.zip?dl=0"
)

LOCAL_DIRECTORY="${1:-"$HOME/load-network"}"

if [ -d "$LOCAL_DIRECTORY" ]; then
    echo "Directory ${LOCAL_DIRECTORY} already exists, not downloading data."
else
    echo "Creating directory: ${LOCAL_DIRECTORY}"
    mkdir -p "${LOCAL_DIRECTORY}"

    echo "Downloading data..."
    pushd "${LOCAL_DIRECTORY}"
    for i in "${!datasets[@]}"
    do
        filename="$i.zip"
        wget -O "$filename" "${datasets[$i]}"
        unzip "$filename"
        rm "${LOCAL_DIRECTORY}/$filename"
    done
    popd

    if [ -d "${LOCAL_DIRECTORY}/__MACOSX" ]; then
        rm -rf "${LOCAL_DIRECTORY}/__MACOSX"
    fi
fi

# if ! type "$nvidia-docker" > /dev/null 2>&1; then
#     echo "You need to install nvidia-docker. See: https://github.com/NVIDIA/nvidia-docker"
# else
#     echo "Getting docker image kaixhin/cuda-torch"
#     nvidia-docker run -it kaixhin/cuda-torch
# fi

if ! type "$th" > /dev/null 2>&1; then
    echo "You need to install torch. See: http://torch.ch/"
else
    pushd load_network_torch
    th test.lua
fi
