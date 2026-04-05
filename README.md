---
title: DC OpenEnv — Data Centre Cooling
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DC OpenEnv (dc-openenv)

OpenEnv environment for **data centre cooling control**: multi-zone thermal dynamics, PUE, carbon-aware signals, and programmatic graders on three difficulty levels. Agents act each step on **fan speeds**, **per-zone supply air setpoints**, and **facility chiller** settings; observations include zone sensors, weather, history, and SLA streaks.

## Action space (`DCAction`)

Defined in [`server/models.py`](server/models.py):

- **`zone_adjustments`**: list of per-zone controls — `zone_id`, `fan_speed_pct` [0, 100], `supply_air_temp_setpoint_c` [16, 26].
- **`chiller_setpoint_c`**: [6, 15] °C.
- **`chiller_active`**: bool.
- **`reasoning`**: optional string (used by the hard-task grader).

## Observation space (`DCObservation`)

Facility-wide fields (time, outdoor temps, chiller state/COP/fault flag, PUE, carbon labels, load phase) plus **`zones`** as `ZoneObservation` records (temperatures, load, humidity, sensor confidence, priority, forecast). Includes a **3-step history** buffer, **SLA violation streak**, and maintenance/event hints.

## Tasks and graders

| Task ID | Difficulty | Max steps | Description |
|---------|------------|-----------|-------------|
| `easy-single-zone` | easy | 48 | Single-zone thermal runaway recovery |
| `medium-multi-zone` | medium | 144 | Multi-zone surge with faulty sensor |
| `hard-cascading-failure` | hard | 288 | Cascading chiller failure + carbon-aware triage |

Step rewards are shaped in roughly **[-1, 1]**; each grader exposes **`final_score()` in [0, 1]** at episode end.

## Setup

```bash
cd /path/to/datacenter-env
uv sync
# optional: dev tools (pytest)
uv sync --extra dev
```

Run the API server locally:

```bash
uv run server
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker (submission / HF)

Build from the **repository root** (required for full `pyproject.toml` + `uv.lock` context):

```bash
docker build -t dc-openenv:latest .
docker run --rm -p 8000:8000 dc-openenv:latest
```

Health: `GET http://localhost:8000/health` — Reset: `POST http://localhost:8000/reset` with JSON body `{}`.

## OpenEnv CLI

```bash
pip install openenv-core   # or use project venv
openenv validate           # run from repo root — should report [OK]
```

## Baseline inference (`inference.py`)

Root-level script using the **OpenAI** Python client against an OpenAI-compatible endpoint.

**Environment variables**

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` or `OPENAI_API_KEY` | Yes | API key |
| `API_BASE_URL` | No | Default: Groq OpenAI-compatible URL |
| `MODEL_NAME` | No | Default model id |
| `INFERENCE_MAX_STEPS_PER_TASK` | No | Cap steps per task (faster smoke runs, &lt;20 min total) |
| `VERBOSE` | No | `1` to print extra `[INFO]` / `[SCORE]` lines |

**Strict stdout protocol** (2 decimal places for `reward`, `score`, and `rewards` CSV):

```text
[START] task=<id> env=dc-openenv model=<model>
[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<str|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1>,<r2>,...
```

By default only these lines are printed (no `[INFO]` noise). Example:

```bash
export HF_TOKEN=...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
uv run python inference.py
```

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
```

## Baseline scores (fill after your run)

| Task | Model | Final score [0,1] |
|------|-------|-------------------|
| easy-single-zone | | |
| medium-multi-zone | | |
| hard-cascading-failure | | |

## Hackathon submission checklist

1. **HF Space** deployed (tagged `openenv`), secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` as required by organizers.
2. **`POST <space-url>/reset`** with `{}` returns **HTTP 200**.
3. **`docker build`** from **this repo root** succeeds.
4. **`openenv validate`** from repo root passes.
5. **`python inference.py`** (or `uv run python inference.py`) completes without error with keys set.
6. Run the organizer **pre-validation script** (ping URL + docker build + `openenv validate`), e.g.:

   ```bash
   ./scripts/validate-submission.sh https://YOUR-SPACE.hf.space /path/to/datacenter-env
   ```

## Project layout

- [`openenv.yaml`](openenv.yaml) — manifest + runtime (`server.app:app`, port 8000)
- [`server/app.py`](server/app.py) — FastAPI app via `create_app`
- [`server/environment.py`](server/environment.py) — `DCEnvironment` (reset / step / state)
- [`server/simulation.py`](server/simulation.py) — physics
- [`server/scenarios/`](server/scenarios/) — task initial conditions
- [`server/graders/`](server/graders/) — per-task scoring
- [`server/client.py`](server/client.py) — `DCEnv` HTTP/WebSocket client
- [`inference.py`](inference.py) — LLM baseline

## License

See `LICENSE` in the repository (BSD-style header in several files).
