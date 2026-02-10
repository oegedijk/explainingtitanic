.ONESHELL:
SHELL := /bin/bash

all: build

build:
	python generate_explainers.py

run:
	gunicorn --preload dashboard:app

fly-launch:
	fly launch --no-deploy

fly-deploy:
	fly deploy

fly-logs:
	fly logs

fly-status:
	fly status

fly-scale-1gb:
	fly scale vm shared-cpu-1x --memory 1024
