#!/usr/bin/bash

LOCAL_TAG=$(date +%Y-%m-%d-%H-%M)
LOCAL_IMAGE_NAME=predict-cancer-model:${LOCAL_TAG}

docker build -t ${LOCAL_IMAGE_NAME} ../dockerfiles