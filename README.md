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

OpenEnv environment for **data centre cooling control**: multi-zone thermal dynamics, physics-based PUE, carbon-aware reward signals, and programmatic graders across three difficulty levels.

Agents control **fan speeds**, **per-zone supply air setpoints**, and a **facility chiller** each step. Observations expose zone sensors, weather conditions, SLA violation streaks, a 3-step history buffer, and event hints.

## MDP definition

| Component | Description |
|-----------|-------------|
| **State space** | Zone temperatures, fan speeds, supply setpoints, chiller COP/fault, PUE, carbon intensity, SLA streak, 3-step history |
| **Action space** | Per-zone `fan_speed_pct` [0,100], `supply_air_temp_setpoint_c` [16,26]; facility `chiller_setpoint_c` [6,15], `chiller_active` bool |
| **Reward** | Shaped [-1, 1]: temperature compliance (primary), PUE vs PID baseline (efficiency), carbon cost, action smoothness, stability bonus |
| **Terminal condition** | Max steps reached, or hard-termination (medium: 10 consecutive SLA violations; hard: critical zone >32 °C for 5+ steps) |
| **Episode horizon** | 48 / 48 / 72 steps for easy / medium / hard |

### Action rate limits (physics)

Actions are clipped by the simulation:
- Fan speed: ±20 % per step
- Supply air setpoint: ±2 °C per step
- Chiller setpoint: ±1 °C per step

## Action space (`DCAction`)

Defined in [`server/models.py`](server/models.py):

- **`zone_adjustments`**: list of per-zone controls — `zone_id`, `fan_speed_pct` [0, 100], `supply_air_temp_setpoint_c` [16, 26].
- **`chiller_setpoint_c`**: [6, 15] °C — facility chilled-water setpoint.
- **`chiller_active`**: bool — turn chiller on/off.
- **`reasoning`**: optional string — graded for coherence on the hard task.

## Observation space (`DCObservation`)

Facility-wide fields:

| Field | Description |
|-------|-------------|
| `timestamp_hour` | Simulated hour [0, 24) |
| `outside_temp_c` | Ambient outdoor temperature |
| `wet_bulb_temp_c` | Wet-bulb temperature (free-cooling headroom) |
| `chiller_active` / `chiller_cop` | Chiller state and coefficient of performance |
| `chiller_fault_detected` | True when COP drops below 60 % of baseline |
| `current_pue` | Power Usage Effectiveness (lower = better) |
| `grid_carbon_intensity` | Categorical: low / medium / high / critical_high |
| `carbon_intensity_normalized` | Continuous [0, 1] |
| `sla_violation_streak` | Consecutive steps with any zone out of [18, 27] °C |
| `history` | Last 3 step snapshots (zone temps, fans, PUE) |
| `zones` | List of `ZoneObservation` (see below) |

Per-zone fields (`ZoneObservation`):

| Field | Description |
|-------|-------------|
| `cold_aisle_temp_c` | True cold-aisle temperature (reliable) |
| `reported_temp_c` | Sensor reading — may be faulty (check `sensor_confidence`) |
| `hot_aisle_temp_c` | Hot-aisle temperature |
| `fan_speed_pct` | Current fan speed |
| `supply_air_temp_c` / `_setpoint_c` | Actual vs setpoint supply air temperature |
| `it_load_kw` / `it_load_pct` | IT load absolute and fraction of base |
| `humidity_pct` | Zone relative humidity |
| `sensor_confidence` | [0, 1] — below 0.5 means reported_temp_c is unreliable |
| `zone_priority` | 2=CRITICAL, 1=MEDIUM, 0=LOW |
| `load_forecast_next_hour` | 1-hour-ahead load forecast in kW |

## Tasks and graders

| Task ID | Difficulty | Max steps | Grader weights | Description |
|---------|------------|-----------|----------------|-------------|
| `easy-single-zone` | easy | 48 | 60% temp compliance + 40% PUE vs baseline | Single zone starts at 28.5 °C (overheating). Recover and maintain. |
| `medium-multi-zone` | medium | 48 | 35% compliance + 25% PUE + 20% sensor inference + 20% peak compliance | 3 zones; one has a faulty sensor reporting +9-12 °C above true temp. |
| `hard-cascading-failure` | hard | 72 | 30% SLA + 25% carbon + 20% recovery + 15% triage + 10% reasoning | Chiller fails at step 20; agent must triage zones by priority. |

### Reward design rationale

- **Temperature compliance is always primary**: PUE reward is suppressed when any zone is out of the [18, 27] °C band — agents should not be penalised for running fans hard during necessary recovery.
- **PUE vs PID baseline** (not vs zero-cooling): reward is relative to a computed PID controller baseline, so moderate improvement scores well without needing near-perfect efficiency.
- **Carbon cost** is a light secondary signal — it encourages shifting cooling intensity away from high-carbon grid windows, but never at the expense of safety.
- **Triage quality** (hard task): post-chiller-failure, agents are rewarded for giving more airflow to CRITICAL zones (`zone_ai_1`, `zone_ai_2`) than to LOW-priority zones (`zone_infra`).
- **Reasoning coherence** (hard task): the grader checks that stated reasoning is consistent with actual actions (e.g., claims triage but sets sacrifice fans higher).

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

Root-level script using the **OpenAI** Python client against any OpenAI-compatible endpoint.

**Environment variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` or `OPENAI_API_KEY` | Yes | — | API key |
| `API_BASE_URL` | No | Groq OpenAI-compatible URL | LLM endpoint |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Model identifier |
| `INFERENCE_MAX_STEPS_PER_TASK` | No | task default | Cap steps per task |
| `VERBOSE` | No | `0` | Set to `1` for `[INFO]`/`[SCORE]` lines |

**Strict stdout protocol** (2 decimal places for `reward`, `score`, and `rewards` CSV):

```text
[START] task=<id> env=dc-openenv model=<model>
[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<str|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1>,<r2>,...
```

By default only these lines are printed. All output is also mirrored to `inference_output.txt`. Example:

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
uv run python inference.py
```

Total wall-clock time: ~5-8 minutes for all three tasks at 0 s inter-step sleep (within the 20-minute hard cap).

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
```

## Baseline scores

| Task | Model | Final score [0, 1] |
|------|-------|--------------------|
| easy-single-zone | llama-3.3-70b-versatile | TBD |
| medium-multi-zone | llama-3.3-70b-versatile | TBD |
| hard-cascading-failure | llama-3.3-70b-versatile | TBD |

## Hackathon submission checklist

1. **HF Space** deployed (tagged `openenv`), secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` as required by organizers.
2. **`POST <space-url>/reset`** with `{}` returns **HTTP 200**.
3. **`docker build`** from **this repo root** succeeds.
4. **`openenv validate`** from repo root passes.
5. **`python inference.py`** (or `uv run python inference.py`) completes without error with keys set.
6. Run the organizer **pre-validation script** (ping URL + docker build + `openenv validate`):

   ```bash
   ./scripts/validate-submission.sh https://YOUR-SPACE.hf.space /path/to/datacenter-env
   ```

## Project layout

- [`openenv.yaml`](openenv.yaml) — manifest + runtime (`server.app:app`, port 8000)
- [`server/app.py`](server/app.py) — FastAPI app via `create_app`
- [`server/environment.py`](server/environment.py) — `DCEnvironment` (reset / step / state)
- [`server/simulation.py`](server/simulation.py) — physics (thermal model, chiller, PUE)
- [`server/scenarios/`](server/scenarios/) — task initial conditions
- [`server/graders/`](server/graders/) — per-task scoring
- [`server/client.py`](server/client.py) — `DCEnv` HTTP/WebSocket client
- [`inference.py`](inference.py) — LLM baseline

## License

See `LICENSE` in the repository (BSD-style header in several files).
