#!/bin/bash

MODEL_REPO_PATH="$(pwd)/.."
MODEL_DIR="clothes_segmentation/1"
MODEL_NAME="densenet_1x3x256x256"
/usr/src/tensorrt/bin/trtexec --onnx=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.onnx \
        --fp16 --saveEngine=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.engine \
        --workspace=1024
