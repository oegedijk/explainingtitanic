.ONESHELL:
SHELL := /bin/bash

all: build

build:
	python generate_explainers.py

run:
	gunicorn --preload dashboard:app