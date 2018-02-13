#!/usr/local/bin/bash

set -e

declare -A datasets=(
    ["scale-domain1"]="https://www.dropbox.com/s/b1ebjekfgkufifh/icw-scale-scale1.zip?dl=0"
    ["scale-domain2"]="https://www.dropbox.com/s/2mt03eguuvbvpe4/icw-scale-scale2.zip?dl=0"

    ["transl-domain1"]="https://www.dropbox.com/s/xv354yvc6orzjeu/icw-translation-left1.zip?dl=0"
    ["transl-domain2"]="https://www.dropbox.com/s/m32dvovkig4b1gn/icw-translation-left2.zip?dl=0"
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
