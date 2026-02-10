---
title: Explaining Titanic
emoji: "🚢"
short_description: Public multi-page ExplainerDashboard demo on Titanic models.
sdk: docker
app_port: 7860
tags:
  - python
  - docker
  - dash
  - explainable-ai
---

# explainingtitanic

Demonstration of the [explainerdashboard](https://github.com/oegedijk/explainerdashboard) package.

This Dash app showcases model quality, permutation importances, SHAP values and interactions, individual trees, and multiple dashboard variants for sklearn-compatible models.

## ExplainerDashboard docs

- Docs: https://explainerdashboard.readthedocs.io
- Example notebook: https://github.com/oegedijk/explainerdashboard/blob/master/dashboard_examples.ipynb

## Local run

With `uv`:

```bash
uv sync
uv run gunicorn --bind 0.0.0.0:7860 dashboard:app
```

Then open `http://localhost:7860`.

## Hugging Face Spaces (Docker)

This repository is configured to run directly as a Hugging Face Docker Space.

- Runtime port: `7860`
- Entrypoint: `gunicorn dashboard:app`
- Health endpoint: `/healthz`

If you duplicate this repo into a Space, set SDK to Docker and deploy.

## Artifact strategy

This demo commits prebuilt explainer artifacts in `pkls/*.joblib` (about 8 MB total).

Why this choice:
- Faster and more reliable cold starts on free CPU Spaces.
- No extra model-building step during container startup.

Tradeoff:
- If model or sklearn versions change, regenerate artifacts with:

```bash
uv run python generate_explainers.py
```

## Other deployment targets

- Fly.io: see `FLY_DEPLOY.md`.
- Heroku compatibility remains via `Procfile` and `.heroku/run.sh`.

