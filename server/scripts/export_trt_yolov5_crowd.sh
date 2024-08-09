#!/bin/bash

MODEL_REPO_PATH="$(pwd)/.."
MODEL_DIR="yolov5_crowd/1"
MODEL_NAME="yolov5_crowd_1x3x384x640"
/usr/src/tensorrt/bin/trtexec --onnx=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.onnx \
        --fp16 --saveEngine=${MODEL_REPO_PATH}/${MODEL_DIR}/${MODEL_NAME}.engine \
        --workspace=1024
        