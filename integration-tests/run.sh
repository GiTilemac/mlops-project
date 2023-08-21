#!/usr/bin/env bash

# Interrupt script on error
set -e

if [[ -z "${GITHUB_ACTIONS}" ]]; then
    # Navigate to this script's directory
    cd "$(dirname "$0")"
fi

# Generate local image using current time tag
# If the image doesn't exist yet, build it
if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=$(date +"%Y-%m-%d-%H-%M")
    export LOCAL_IMAGE_NAME="predict-cancer-model:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t "${LOCAL_IMAGE_NAME}" ../dockerfiles
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

# Start the prediction container via docker-compose
docker-compose up -d

sleep 5 # Sleep more, to give container time to start

# Send the request to the service and test the response
python test_docker.py

# Print logs and exit on error
ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

# Shut the container down
docker-compose down
