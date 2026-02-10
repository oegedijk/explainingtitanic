# Fly.io deployment guide

This repository keeps Heroku deployment compatibility and adds Fly.io deployment as an additive path.

## 1) One-time setup

Install tools:

```bash
brew install flyctl uv just
```

Authenticate Fly:

```bash
fly auth login
fly auth whoami
```

Install/sync Python dependencies with `uv`:

```bash
uv sync
```

If you change `scikit-learn` versions, regenerate serialized artifacts so runtime and model files stay compatible:

```bash
uv run python generate_explainers.py
```

## 2) Public-repo security baseline

Never commit secrets. This repo is public.

Set only deployment/admin secrets in Fly (example):

```bash
fly secrets set \
  FLY_API_TOKEN="replace-with-ci-token-if-needed"
```

Notes:
- Keep `.env` and `.env.*` local only.
- Rotate secrets if they are ever exposed.
- Prefer Fly secrets over hardcoded values in `fly.toml` or committed files.
- This demo intentionally has no app-level auth (public read access).

## 3) Create app config (first time only)

From repo root:

```bash
fly launch --no-deploy
```

Suggested answers:
- Reuse existing `fly.toml`: `yes`
- Region: choose one near users
- Postgres/Redis: `no`

If app name is taken, update `app = "explainingtitanic"` in `fly.toml` to a unique name.

## 4) Deploy

```bash
fly deploy
fly open
```

## 5) Day-2 operations

Status:

```bash
fly status
```

Logs:

```bash
fly logs
```

Restart all machines:

```bash
for id in $(fly machine list --json | jq -r '.[].id'); do fly machine restart "$id"; done
```

## 6) Scale CPU/RAM

Default profile in `fly.toml`: shared CPU, 1GB RAM.

Explicitly set it:

```bash
fly scale vm shared-cpu-1x --memory 1024
```

Lower-cost option (if stable for your load):

```bash
fly scale vm shared-cpu-1x --memory 512
```

## 7) Rollback

```bash
fly releases
fly releases rollback
# or
fly releases rollback <VERSION>
```

## 8) Cost behavior notes

- `auto_stop_machines = "suspend"` and `min_machines_running = 0` minimize idle compute cost.
- `auto_start_machines = true` wakes the app on demand.
- First request after idle can be slower due to cold start.
- Final cost still depends on plan, bandwidth, and any additional Fly resources.

### Startup-latency tuning (measured)

Current runtime defaults:
- `GUNICORN_PRELOAD=0`
- Health check grace period: `45s`

Measured cold-start probes on `ord` (1GB shared CPU):
- `preload=0`: `/healthz` ready in `28-56s`; first `/classifier/` in `1.2-12.4s`
- `preload=1`: `/healthz` ready in `28-60s`; first `/classifier/` in `1.4-14.7s`

Conclusion:
- Keep `preload=0` (no consistent win from preload; lower memory pressure risk).
- Keep health-check grace at `45s` to avoid false negatives during cold boot.

Optional keep-warm strategy (faster first-hit, higher cost):
- Keep one machine always running by setting:
  - `auto_stop_machines = "off"`
  - `min_machines_running = 1`
- Re-deploy after config change:
```bash
fly deploy
```

## 9) Local container validation

Build:

```bash
docker build -t explainingtitanic:fly .
```

Run:

```bash
docker run --rm -p 8080:8080 explainingtitanic:fly
```

Check:

```bash
curl -i http://localhost:8080/
```

Expected: HTTP `200`.

## 10) Optional `just` shortcuts

```bash
just sync
just lock
just export-requirements
just fly-launch
just fly-deploy
just fly-logs
just fly-status
just fly-scale-1gb
```

## 11) Optional warmup proxy app

To serve a quick "waking up" page while the main app is suspended, deploy the `proxy/` app:

```bash
cd proxy
fly launch --no-deploy --name titanicexplainer --copy-config
fly deploy
fly open
```

Behavior:
- `https://titanicexplainer.fly.dev` proxies to `https://explainingtitanic.fly.dev`.
- If backend is cold/unreachable, users see a warmup page that auto-retries.
- Proxy app is configured as tiny always-on (`shared-cpu-1x`, `256MB`) for fast first paint.
