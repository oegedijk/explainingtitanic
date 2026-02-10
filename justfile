set dotenv-load := false

default:
  @just --list

sync:
  uv sync

lock:
  uv lock

export-requirements:
  uv export --no-dev --format requirements-txt -o requirements.txt

run:
  uv run gunicorn --preload dashboard:app

docker-build:
  docker build -t explainingtitanic:fly .

docker-run:
  docker run --rm -p 8080:8080 explainingtitanic:fly

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

proxy-deploy:
  cd proxy && fly deploy -a titanicexplainer

proxy-status:
  fly status -a titanicexplainer

proxy-logs:
  fly logs -a titanicexplainer
