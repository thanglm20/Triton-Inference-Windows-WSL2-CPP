#!/bin/bash

MODEL_REPO_PATH="$(pwd)/.."
MODEL_DIR="clothes_segmentation/1"
MODEL_NAME="densenet_1x3x256x256"
INPUT_SIZE=256
MIN_BATCH=1
MAX_BATCH=64
/usr/src/tensorrt/bin/trtexec --onnx=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.onnx \
        --minShapes=input:${MIN_BATCH}x3x${INPUT_SIZE}x${INPUT_SIZE} \
        --optShapes=input:8x3x${INPUT_SIZE}x${INPUT_SIZE} \
        --maxShapes=input:${MAX_BATCH}x3x${INPUT_SIZE}x${INPUT_SIZE} \
        --fp16 --saveEngine=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.engine \
        --workspace=1024
