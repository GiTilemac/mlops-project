LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=predict-cancer-model:${LOCAL_TAG}

test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} ./dockerfiles

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash ./integration-tests/run.sh
