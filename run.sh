#!/usr/local/bin/bash

set -e

declare -A datasets=(
    ["scale-domains"]="https://www.dropbox.com/s/mvwv3idkkde6y3s/icw-scale-domains.zip?dl=0"
    ["scale-domain1"]="https://www.dropbox.com/s/2kyjq50b07zjcc8/icw-scale-scale1.zip?dl=0"
    ["scale-domain2"]="https://www.dropbox.com/s/2wnqurtk8iczu7c/icw-scale-scale2.zip?dl=0"

    ["transl-domains"]="https://www.dropbox.com/s/ftm1a3r2zzyexyr/icw-translation-domains.zip?dl=0"
    ["transl-domain1"]="https://www.dropbox.com/s/7axg27opvdr2dxv/icw-translation-left1.zip?dl=0"
    ["transl-domain2"]="https://www.dropbox.com/s/ebjuunmiwu53o0f/icw-translation-left2.zip?dl=0"
)

LOCAL_DIRECTORY="${1:-"$HOME/load-network"}"

if [ -d "$LOCAL_DIRECTORY" ]; then
    echo "Directory ${LOCAL_DIRECTORY} already exists, not downloading data."
else
    echo "Creating directory: ${LOCAL_DIRECTORY}"
    mkdir "${LOCAL_DIRECTORY}"

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

    rm -r "${LOCAL_DIRECTORY}/__MACOSX"
fi
