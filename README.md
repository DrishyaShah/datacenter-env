---
title: RL Environment for Datacenter Cooling and Operations
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# RL Environment for Datacenter Cooling and Operations

**Themes:** Multi-Agent Interactions (Theme #1) · World Modeling / Professional Tasks (Theme #3.1) — Meta × HuggingFace × Scaler OpenEnv Hackathon Finale 2026

| | |
|---|---|
| **HF Space (live environment)** | [Mephisto2412/datacenter-env](https://huggingface.co/spaces/Mephisto2412/datacenter-env) |
| **PPO Cooling Controller** | [Mephisto2412/clusterenv-ppo-cooling](https://huggingface.co/Mephisto2412/clusterenv-ppo-cooling) |
| **GRPO Adapter** | [Mephisto2412/clusterenv-grpo-adapter](https://huggingface.co/Mephisto2412/clusterenv-grpo-adapter) |
| **Training Notebook (Colab)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/main/training/train_grpo_colab.ipynb) |
| **Training Logs (HF Space 50-iter)** | [training_logs_hfspace_50iter.txt](training/training_logs_hfspace_50iter.txt) |
| **Mini-Blog** | [BLOG.md](BLOG.md) |
| **GitHub Repo** | [DrishyaShah/datacenter-env](https://github.com/DrishyaShah/datacenter-env/tree/main) |

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Why Two Agents?](#why-two-agents)
3. [System Architecture](#system-architecture)
4. [Physics Engine](#physics-engine)
5. [Datacenter Cooling Tasks](#datacenter-cooling-tasks)
   - [Easy: Single-Zone Thermal Runaway Recovery](#easy-single-zone-thermal-runaway-recovery)
   - [Medium: Multi-Zone Load Surge with Sensor Fault](#medium-multi-zone-load-surge-with-sensor-fault)
   - [Hard: Cascading Chiller Failure with Carbon-Aware Triage](#hard-cascading-chiller-failure-with-carbon-aware-triage)
   - [Timeline Condensation](#timeline-condensation)
   - [Cooling Observation Space](#cooling-observation-space)
   - [Cooling Action Space](#cooling-action-space)
   - [Cooling Reward Functions](#cooling-reward-functions)
   - [LLM Cooling Agent](#llm-cooling-agent)
6. [ClusterEnv: Scheduling Layer](#clusterenv-scheduling-layer)
   - [Episode Structure](#episode-structure)
   - [Scheduling Observation Space](#scheduling-observation-space)
   - [Scheduling Action Space](#scheduling-action-space)
   - [The Two Teams](#the-two-teams)
   - [Oversight Monitor](#oversight-monitor)
   - [Scheduling Reward Function](#scheduling-reward-function)
7. [PPO Cooling Controller Training](#ppo-cooling-controller-training)
8. [GRPO Training](#grpo-training)
9. [Training Results](#training-results)
10. [How to Run](#how-to-run)
11. [Design Decisions and Caveats](#design-decisions-and-caveats)
12. [File Reference](#file-reference)

---

## The Problem

A datacenter is a physical system and an economic system simultaneously, and decisions in one layer propagate directly into the other. On the physical side, servers generate heat continuously — cooling systems must remove it in real time while managing power consumption and carbon output. Cooling is not a static setpoint problem: the right control action depends on current zone temperatures, IT load, outside conditions, chiller efficiency, and the thermal inertia of the facility. Equipment fails mid-episode. Sensors drift. The system that was in equilibrium at noon may be approaching a thermal limit by 2pm.

On the operational side, compute is shared infrastructure. Multiple teams submit jobs with stated priorities, deadlines, and resource requirements. Those jobs, once admitted, become IT load — directly affecting the physical layer. A scheduling decision is not just an allocation decision; it is a thermal and electrical commitment that the physical system must absorb for the duration of the job's runtime.

Carbon adds a third dimension. Grid emissions intensity is not constant — it tracks the generation mix, which shifts over hours as demand rises and renewable sources cycle in and out. Many compute workloads are temporally flexible: they have real deadlines, but those deadlines carry slack. A scheduler with visibility into the carbon forecast and genuine knowledge of which jobs are deferrable can systematically route flexible work toward lower-emission windows. The difficulty is that this requires accurate information about job flexibility — and in shared infrastructure, teams have structural incentives to misrepresent exactly that.

When compute is a constrained shared resource, teams benefit from claiming higher priority than warranted, asserting tighter deadlines than exist, and concealing flexibility that would allow their jobs to be deferred. Each misrepresentation is individually rational: it improves a team's allocation outcome. Collectively, they degrade the scheduler's ability to make good decisions — over-allocating to aggressive claimants, crowding out legitimate work, missing carbon deferral opportunities, and potentially exceeding facility power budgets.

This environment models the full stack: the thermal physics of a multi-zone facility under variable load; the economic layer of job admission, chargeback, and team budgets; the adversarial dynamic where one team systematically misrepresents its job metadata; and an oversight layer that detects gaming patterns and feeds them back to the scheduler. Two AI agents operate across these layers — a PPO controller for the physical layer, an LLM scheduler for the operational layer — each handling the problem it is built for.

---

## Why Two Agents?

### Why RL (PPO) for cooling

Datacenter cooling is a **continuous numeric control problem**. The action at each step — fan speed percentage, chilled-water supply setpoint — is determined by physical state: zone temperatures, IT load, outside conditions, chiller efficiency. The correct response is largely computable from the physics, and the same corrective logic applies every step. There are no natural language descriptions to parse, no competing contextual claims to weigh, no multi-step reasoning to do. What the task needs is a policy that can map sensor readings to control actions thousands of times per episode without any latency overhead.

PPO handles this well. It learns a compact MLP policy that encodes the physics implicitly through training: it discovers that rising temperatures require more fan speed, that pre-cooling before a large job lands saves energy later, that free-cooling windows reduce chiller load. It does this through environment feedback alone, no reward engineering for individual behaviours. Once trained, it runs in microseconds per step — essential for a simulation running 18 physical steps per scheduling window.

An LLM doing this same job would call an API 18 × 8 = 144 times per episode. The API latency alone would dominate wall-clock time. More importantly, the cooling control problem gives the model nothing to do with its language capabilities — it would just be a slow, expensive lookup table. PPO is not a fallback here; it is the correct tool.

### Why an LLM for scheduling

The scheduling problem is structurally different from the cooling problem in every relevant dimension.

Job requests arrive as natural language descriptions carrying stated priorities, deadlines, and carbon preferences. A correct admission decision requires cross-referencing those claims against team history, current power headroom, the carbon forecast, and whatever the oversight monitor flagged about this team in the previous window. The relevant signals are heterogeneous — some are structured numerics, some are free-text descriptions, some are patterns in historical rates. Collapsing them into a fixed feature vector and training a classifier would work only if the mapping from inputs to correct decisions were stable. It is not: the correct interpretation of a stated "CRITICAL" priority depends on whether that team has inflated priority on 80% of its previous submissions or 10%.

The scheduler acts once per window — eight times per episode. That is not a latency-sensitive workload. One LLM inference call per window is entirely practical.

The action space is itself structured language: the model reads a JSON observation, produces a JSON list of `ACCEPT` / `REJECT` / `DEFER` decisions per job, and the environment scores each outcome against a verifiable reward signal. This is the setting GRPO was designed for — reward-verifiable decisions with no labeled demonstrations available.

The deeper reason this decomposition is correct rather than indulgent: the rule-based baseline scheduler demonstrates exactly where tabular policies break down. It sorts by `stated_priority`, accepts at 85% capacity, and ignores carbon entirely. Against Team B, which always claims `HIGH` or `CRITICAL` regardless of true priority, it consistently fills capacity with inflated-priority jobs before honest ones reach the front of the queue. The failure is not a matter of the baseline being poorly tuned — it is a structural limitation of any policy that treats stated metadata as ground truth. Correcting for systematic misrepresentation requires maintaining beliefs about a team's honesty, updating those beliefs across windows, and weighing stated claims accordingly. That is reasoning over context, and language models are the appropriate mechanism for it.

The question the environment poses is whether a 3B-parameter model can learn this from reward alone — no labeled demonstrations, no oracle policy to imitate, just the signal from scheduling outcomes across episodes. The training results show it can, and that the convergence is stable.

---

## System Architecture

![ClusterEnv System Architecture](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/main/training/system-architecture.png)

*Team A and Team B submit job requests to the LLM Scheduler (Qwen2.5-3B, GRPO-trained), which issues ACCEPT/REJECT/DEFER decisions per window. Admitted jobs flow through the Economic Layer to the PPO Cooling Controller (SB3, pre-trained), which runs 18 physical simulation steps per window. The OversightMonitor compares stated vs. ground-truth metadata and injects gaming flags into the next window's observation. The window reward (50% throughput + 35% thermal + 15% carbon) closes the GRPO training loop.*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ClusterEnvironment                                  │
│                                                                                │
│  Team A (honest)  →  ┌──────────────────┐   WindowState (JSON prompt)        │
│  Team B (gaming)  →  │  LLM Scheduler   │ ←──────────────────────────────    │
│                       │  Qwen2.5-3B      │ ──────────────────────────────→    │
│                       │  GRPO / LoRA r=16│   AdmissionDecision × N            │
│                       └────────┬─────────┘   (ACCEPT / REJECT / DEFER)        │
│                                │                                               │
│                                ▼                                               │
│               ┌──────────────────────────────────────────┐                    │
│               │              Economic Layer               │                    │
│               │  ChargebackLedger  ·  EpisodeLedger       │                    │
│               │  OversightMonitor (4 detectors)           │                    │
│               └──────────────────┬───────────────────────┘                    │
│                                   │ IT load injected into zones                │
│                                   ▼                                            │
│               ┌──────────────────────────────────────────────────────────┐    │
│               │      Physical Layer — 18 steps × 5 min per window        │    │
│               │                                                            │    │
│               │  PPO Cooling Controller  →  FacilityState.step()         │    │
│               │  (SB3 MLP, pre-trained)      zone_team_a_1/2              │    │
│               │  fan: [0–100%]               zone_team_b_1                │    │
│               │  setpoint: [6–26°C]          zone_shared                  │    │
│               └──────────────────────────────────────────────────────────┘    │
│                                   │                                            │
│                                   ▼                                            │
│               ┌──────────────────────────────────────────────────────────┐    │
│               │  ClusterGrader.record_window()                            │    │
│               │  R = 0.50 × throughput + 0.35 × thermal + 0.15 × carbon  │    │
│               └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘

Data flow per window:
  Teams generate requests → WindowState built → LLM issues decisions
  → Admitted jobs set zone IT loads → PPO runs 18 physics steps
  → OversightMonitor analyzes requests → flags injected into next WindowState
  → ClusterGrader records window reward → episode advances
```

The same `FacilityState` physics engine also backs three standalone cooling tasks — Easy, Medium, and Hard — evaluated separately against a direct LLM cooling agent (`inference.py`). Those tasks are documented in [Datacenter Cooling Tasks](#datacenter-cooling-tasks).

---

## Physics Engine

All thermal physics live in `server/simulation.py`. This engine is shared across the three cooling tasks and the ClusterEnv scheduling environment.

### Zone thermal model

Each `ZoneState` has a configurable thermal mass (`thermal_mass_kj_per_k`, default 850 kJ/K, scaled proportionally to zone IT load). Temperature update at each step:

```
heat_in   = it_load_kw × SECONDS_PER_STEP (300 s)
heat_out  = mass_flow × Cp_air × (zone_temp − supply_air_temp)
ΔT        = (heat_in − heat_out) × SECONDS_PER_STEP / (thermal_mass_kj_per_k × 1000)
zone.temp += ΔT
```

Where `mass_flow` scales with `fan_speed_pct` and zone capacity:

```
mass_flow      = (fan_speed_pct / 100) × MASS_FLOW_REF_KGS × capacity_ratio
capacity_ratio = zone.cooling_capacity_kw / MASS_FLOW_REF_CAPACITY_KW
```

The cold-aisle temperature floor is clamped to prevent physically impossible sub-ambient values.

### Chiller and free cooling

- **Chiller COP** is temperature-dependent: warmer outside air reduces efficiency. COP degrades approximately linearly from 3.5 at 20°C as `outside_temp_c` rises.
- **Free cooling** (`free_cooling_potential`) measures how much cooling could be supplied by outside-air economiser. Active only when `wet_bulb_temp_c` is meaningfully below the target supply temperature. The chiller propagation logic blends free-cooling air only when it is genuinely cooler than the chilled-water target.
- **Chiller fault**: `chiller_fault_step` triggers COP degradation from 3.5 → 0.8 over 5 steps, followed by full offline state. Observable via `chiller_fault_detected` flag (set when COP < 60% of baseline) in the cooling tasks, and via `thermal_summary` color shifts in ClusterEnv.

### Diurnal curves

The Medium and Hard cooling tasks provide per-step outside temperature and wet-bulb curves (144 and 288 raw data points respectively). The environment uses `step_scale` to index into these curves at the condensed rate. IT load follows a 24-hour sinusoidal/trapezoidal profile. Carbon intensity follows a separate 24-hour curve with peak midday values.

### Sensor drift (Medium cooling task)

`zone_ai` has a sensor fault. `apply_sensor_drift()` accumulates drift using an effective step count scaled by `minutes_per_step / 5.0`:

```
effective_step = raw_step × (minutes_per_step / 5.0)
target_drift   = min(3.0 + effective_step × 0.18, 12.0)   # caps at +12°C
```

`reported_temp_c` includes this drift. `sensor_confidence` degrades from 1.0 → ~0.1 as drift accumulates. `cold_aisle_temp_c` always shows the true physical temperature.

### Rate limiting on actions

`simulation.step()` applies soft rate limiting: consecutive large fan speed or setpoint changes are partially smoothed to prevent instantaneous step changes that would be physically unrealistic.

---

## Datacenter Cooling Tasks

The environment exposes three standalone cooling tasks evaluated against a direct LLM cooling agent (`inference.py`). Each task gives the agent full control over fan speeds and chiller setpoints, and scores it on temperature compliance, energy efficiency, and carbon awareness.

### Easy: Single-Zone Thermal Runaway Recovery

```
┌──────────────────────────────────────────────┐
│              Data Centre (Easy)               │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │               zone_main                 │  │
│  │  Priority: MEDIUM                       │  │
│  │  IT load:  450 kW (constant)            │  │
│  │  Start T:  28.5°C  ← OVERHEATING        │  │
│  │  Target:   [18–27°C]                    │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│  Outside: 32°C (hot summer afternoon)         │
│  Chiller: available, no faults               │
│  Grid:    medium carbon                       │
│  Time:    14:00 → 18:00 (4 hours)            │
└──────────────────────────────────────────────┘

Episode: 20 steps × 12 min/step (step_scale=2.4)
```

**Goal**: Cool the overheating zone into `[18, 27]°C`, then maintain it efficiently. Not just pinning fans at 100% — the agent must back off once the zone is stable to earn PUE reward.

**Hard termination**: none.

**Final score**: `0.60 × compliance_fraction + 0.40 × avg_pue_score`

---

### Medium: Multi-Zone Load Surge with Sensor Fault

```
┌────────────────────────────────────────────────────────┐
│                 Data Centre (Medium)                    │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │  zone_ai             │  │  zone_storage        │    │
│  │  Priority: CRITICAL  │  │  Priority: MEDIUM    │    │
│  │  IT load: 600 kW     │  │  IT load: 200 kW     │    │
│  │  FAULTY SENSOR ⚠     │  │  (no fault)          │    │
│  │  sensor_confidence   │  └──────────────────────┘    │
│  │  degrades 1.0→0.1    │                              │
│  │  by step ~10         │  ┌──────────────────────┐    │
│  └──────────────────────┘  │  zone_infra          │    │
│                             │  Priority: LOW       │    │
│                             │  IT load: 150 kW     │    │
│                             └──────────────────────┘    │
│                                                         │
│  Outside: 18°C (night) → 34°C peak (noon)              │
│  Load surge: steps 6–17 (~60% → ~95% of baseline)      │
│  Time: 06:00 → 18:00 (12 hours)                        │
└────────────────────────────────────────────────────────┘

Episode: 30 steps × 24 min/step (step_scale=4.8)
```

**Goal**: Keep all three zones in `[18, 27]°C` through a load surge while `zone_ai`'s sensor drifts up to +12°C above the true temperature. The agent must use `cold_aisle_temp_c` and `sensor_confidence` to infer true state — trusting `reported_temp_c` leads to systematic over-cooling.

**Hard termination**: any zone unsafe for 10+ consecutive steps → episode ends with score 0.

**Final score**:
```
0.35 × all_zone_compliance_fraction
+ 0.25 × avg_pue_score
+ 0.20 × sensor_inference_quality
+ 0.20 × peak_window_compliance (steps 6–17)
```

Sensor inference quality is scored at episode end by comparing the agent's `supply_air_temp_setpoint_c` for `zone_ai` against an oracle setpoint (20°C during high load, 22°C normal), averaged over steps when `sensor_confidence < 0.5`.

---

### Hard: Cascading Chiller Failure with Carbon-Aware Triage

```
┌──────────────────────────────────────────────────────────────────┐
│                     Data Centre (Hard)                            │
│                                                                    │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │  zone_ai_1       │  │  zone_ai_2       │  ← CRITICAL           │
│  │  Priority: 2     │  │  Priority: 2     │    must stay ≤ 30°C   │
│  │  500 kW          │  │  480 kW          │                        │
│  └──────────────────┘  └──────────────────┘                       │
│                                                                    │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │  zone_storage    │  │  zone_infra      │  ← Sacrificeable       │
│  │  Priority: 1     │  │  Priority: 0     │    (LOW)               │
│  │  200 kW          │  │  120 kW          │                        │
│  └──────────────────┘  └──────────────────┘                       │
│                                                                    │
│  Chiller fault timeline:                                          │
│    Steps 0–2  : Normal operation (COP ≈ 3.5)                     │
│    Step 3     : COP begins degrading → 0.8 over 5 steps          │
│    Step 8     : Chiller OFFLINE — fans only from here            │
│    Steps 8–16 : Recovery window (free cooling + fans only)       │
│                                                                    │
│  Carbon: low nights → HIGH midday (steps 11–22) → low evening    │
│  Free cooling: steps 0–4 and ~33–40 (cool night air)             │
│  Time: 00:00 → 24:00 (24 hours)                                  │
└──────────────────────────────────────────────────────────────────┘

Episode: 40 steps × 36 min/step (step_scale=7.2)
```

**Goal**: Protect the critical AI zones through a chiller failure. Pre-cool before the fault is confirmed, triage resources post-fault, exploit free-cooling windows, and avoid running full fans during high-carbon midday.

**Hard termination**: any critical zone above 32°C for 5+ consecutive steps → episode ends with score 0.

**Final score**:
```
0.30 × sla_score         (critical zone safety throughout)
+ 0.25 × carbon_score    (efficiency during high-carbon window, steps ~11–22)
+ 0.20 × recovery_score  (critical zones in safe band during steps 8–16)
+ 0.15 × triage_score    (prioritising critical zones over low-priority)
+ 0.10 × reasoning_score (stated reasoning matches actual action)
```

**Hard termination** detail: if `chiller_active=False` is observed and any critical zone exceeds 32°C for 5+ consecutive steps, the episode terminates with `score=0`. Once the chiller goes offline at step 8, setting `chiller_active=True` in the action has no effect — the simulation ignores it.

---

### Timeline Condensation

The original scenario plans span 48 / 144 / 288 steps at 5 min/step. To fit within the ~20-minute inference budget, episodes are condensed:

| Task | Original steps | Condensed steps | `step_scale` | Sim time per step | Total simulated |
|------|---------------|-----------------|--------------|-------------------|-----------------|
| Easy | 48 | 20 | 2.4 | 12 min | 4 hr |
| Medium | 144 | 30 | 4.8 | 24 min | 12 hr |
| Hard | 288 | 40 | 7.2 | 36 min | 24 hr |

**How it works**: `minutes_per_step = 5.0 × step_scale` is stored in `FacilityState`. Each step:

1. **Clock** advances by `minutes_per_step` (e.g. 36 min for Hard).
2. **Weather curves** are indexed at `step_count × step_scale` to traverse the full arc.
3. **Load and carbon curves** follow the clock (hour-indexed) at the correct condensed rate.
4. **Sensor drift** uses `effective_step = raw_step × step_scale` so drift speed is proportionally correct.
5. **Chiller fault step** is rescaled on `reset()`: `scaled_fault = round(raw_fault_step / step_scale)`.
6. **Thermal physics** (`step_thermal()`) still uses `SECONDS_PER_STEP = 300` (5 real minutes) to keep heat-transfer calculations physically accurate.

The result: the agent experiences the full scenario arc (night → morning surge → peak → recovery) within a tractable step count, while individual temperature changes per step remain realistic.

---

### Cooling Observation Space

Returned as a `DCObservation` Pydantic model each step. All fields present for all tasks.

#### Facility-level fields

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current step number (0-indexed) |
| `timestamp_hour` | `float` | Hour of day [0–24] |
| `timestamp_day_sin` | `float` | sin(2π × hour / 24) — cyclical time encoding |
| `timestamp_day_cos` | `float` | cos(2π × hour / 24) — cyclical time encoding |
| `outside_temp_c` | `float` | Outdoor dry-bulb temperature (°C) |
| `wet_bulb_temp_c` | `float` | Outdoor wet-bulb temperature (°C) — determines free-cooling potential |
| `chiller_active` | `bool` | Whether the chiller is currently running |
| `chiller_setpoint_c` | `float` | Current chilled-water setpoint [6–15] (°C) |
| `chiller_cop` | `float` | Current chiller coefficient of performance |
| `chiller_fault_detected` | `bool` | True when COP < 60% of baseline or chiller is offline |
| `ups_efficiency` | `float` | UPS efficiency [0–1] |
| `current_pue` | `float` | Real-time Power Usage Effectiveness (1.0 = perfect) |
| `free_cooling_potential` | `float` | Fraction of cooling that could be met by free-air economiser [0–1] |
| `grid_carbon_intensity` | `str` | `"low"` / `"medium"` / `"high"` / `"critical_high"` |
| `carbon_intensity_normalized` | `float` | Numeric carbon intensity [0.0–1.0] |
| `load_curve_phase` | `str` | `"ramp_up"` / `"peak"` / `"ramp_down"` / `"idle"` |
| `sla_violation_streak` | `int` | Consecutive steps where any zone was outside [18, 27]°C |
| `maintenance_active` | `bool` | True if any zone is in a maintenance window |
| `maintenance_notes` | `list[str]` | Free-text maintenance notes |
| `upcoming_events` | `list[str]` | Scenario-injected event forecasts |
| `history` | `list[dict]` | Last 3 step snapshots (per-zone temps, fan speed, PUE) — oldest first |

#### Per-zone fields (`zones` array)

Each entry is a `ZoneObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `zone_id` | `str` | Zone identifier (e.g. `zone_main`, `zone_ai_1`) |
| `cold_aisle_temp_c` | `float` | **True** cold-aisle supply temperature (°C) — always accurate |
| `hot_aisle_temp_c` | `float` | Return-air temperature from server exhausts (°C) |
| `reported_temp_c` | `float` | Sensor-reported temperature — **may include drift/fault offset** |
| `supply_air_temp_c` | `float` | Actual delivered supply air temperature after chiller blending |
| `supply_air_temp_setpoint_c` | `float` | Agent-controlled supply air setpoint [16–26] (°C) |
| `it_load_kw` | `float` | Current IT equipment power draw (kW) |
| `it_load_pct` | `float` | Normalised IT load relative to zone baseline [0–1] |
| `fan_speed_pct` | `float` | Current fan speed [0–100%] |
| `cooling_capacity_kw` | `float` | Maximum cooling capacity at full fan speed (kW) |
| `humidity_pct` | `float` | Relative humidity (%) |
| `sensor_confidence` | `float` | Reliability weight [0.0–1.0]; below 0.5 means `reported_temp_c` is unreliable |
| `zone_priority` | `int` | Static criticality: 0=LOW, 1=MEDIUM, 2=CRITICAL |
| `load_forecast_next_hour` | `float` | Predicted IT load 60 min ahead (kW) |

#### Dynamic alerts (injected by `inference.py`)

`inference.py` computes additional real-time warnings via `_compute_alerts()` and injects them as an `alerts` list into the JSON prompt. These are not part of `DCObservation` — they are added only to the LLM's prompt context:

- Chiller fault / offline warnings
- Zone over-temperature warnings with delta trend
- Sensor fault warnings when `sensor_confidence < 0.4`
- Carbon intensity warnings during high-carbon windows
- SLA violation streak warnings
- Efficiency nudge when a zone is stable but fans are unnecessarily high (≥70% fan with safe, stable temperature)

---

### Cooling Action Space

Submitted as a `DCAction` JSON object each step.

#### Per-zone adjustments (`zone_adjustments` array)

| Field | Type | Bounds | Description |
|-------|------|--------|-------------|
| `zone_id` | `str` | — | Must exactly match a `zone_id` from the current observation |
| `fan_speed_pct` | `float` | [0.0, 100.0] | Target fan speed for this zone |
| `supply_air_temp_setpoint_c` | `float` | [16.0, 26.0] | Target supply air temperature setpoint |

#### Facility-level controls

| Field | Type | Bounds | Default | Description |
|-------|------|--------|---------|-------------|
| `chiller_setpoint_c` | `float` | [6.0, 15.0] | 10.0 | Facility-wide chilled-water setpoint |
| `chiller_active` | `bool` | — | true | Toggle chiller on/off |
| `reasoning` | `str` | — | null | Agent's explanation; graded in Hard task for coherence |

**Rate limiting**: the simulation smooths abrupt consecutive changes. Very large single-step jumps are partially applied rather than fully accepted, reflecting real actuator dynamics.

**Omitting a zone** from `zone_adjustments` leaves its settings unchanged for that step.

---

### Cooling Reward Functions

All rewards are per-step values clipped to `[−1.0, 1.0]`. Each grader also produces a `final_score ∈ [0.0, 1.0]` at episode end.

#### Easy task (`grader_easy.py`)

```
If zone in [18, 27]°C:
  closeness       = 1.0 - |temp - 22| / 5.0                    (0→1)
  dist_boundary   = min(temp - 18, 27 - temp)
  boundary_margin = min(dist_boundary / 3.0, 1.0)              (0→1)
  temp_reward     = 0.30 + 0.10 × closeness + 0.15 × boundary_margin
  streak_bonus    = 0.05 × min(consecutive_safe / 10, 1.0)
  temp_reward     = min(0.60, temp_reward + streak_bonus)

  pue_vs_pid  = (pid_baseline_pue - current_pue) / (pid_baseline_pue - 1.18)
  pue_reward  = 0.35 × pue_vs_pid   (clamped to [-1, 1])

Else (violation):
  overshoot   = max(0, temp - 27)
  undershoot  = max(0, 18 - temp)
  temp_reward = -0.30 × min((overshoot + undershoot) / 3.0, 1.0)
  pue_reward  = 0.0   (suppressed during violation)

carbon_reward = -0.05 × cooling_overhead_fraction × carbon_normalized

R_step = clip(temp_reward + pue_reward + carbon_reward, -1, 1)
```

**Final score**: `0.60 × compliance_fraction + 0.40 × avg_pue_score`

#### Medium task (`grader_medium.py`)

**Per-step reward weights**: temp=0.50, PUE=0.25, carbon=0.15, roughness=0.10

Priority multipliers on temperature reward: LOW=0.7×, MEDIUM=1.0×, CRITICAL=1.4×

**Sensor inference quality**: scored at episode end by comparing agent's `supply_air_temp_setpoint_c` for `zone_ai` against an oracle setpoint (20°C during high load, 22°C normal), averaged over steps when `sensor_confidence < 0.5`. Rewards agents that act on true physical temperature rather than the drifted sensor reading.

**Final score**:
```
0.35 × all_zone_compliance
+ 0.25 × avg_pue_score
+ 0.20 × sensor_inference_quality
+ 0.20 × peak_window_compliance  (steps 6–17)
```

#### Hard task (`grader_hard.py`)

**Per-step reward weights**: temp=0.45, PUE=0.20, carbon=0.05, safety=0.20, roughness=0.05, stability=0.05

**SLA compliance**: critical zones above `CRITICAL_THRESHOLD=30°C` incur hard safety penalties. Above `EMERGENCY_THRESHOLD=35°C` = maximum penalty.

**Triage quality**: at each step after `CHILLER_OFFLINE_STEP=8`, checks whether critical zones are being prioritised (higher fan, lower setpoint) relative to low-priority `zone_infra`.

**Recovery speed**: fraction of steps in the recovery window `[8, 16]` where all critical zones are in safe band `[18, 27]°C`.

**Carbon efficiency**: fraction of high-carbon steps (where `carbon_intensity_normalized > 0.55`) where cooling power is below median. Defaults to 0.5 if the episode never reaches a high-carbon window.

**Reasoning coherence**: regex-scored against declared crisis actions. An agent that says "raising fans to protect critical zones" but actually lowers fans loses coherence points.

**Final score**:
```
0.30 × sla_score
+ 0.25 × carbon_score
+ 0.20 × recovery_score
+ 0.15 × triage_score
+ 0.10 × reasoning_score
```

**Success thresholds** (calibrated to task difficulty):
- Easy: ≥ 0.55
- Medium: ≥ 0.50
- Hard: ≥ 0.40

---

### LLM Cooling Agent

The cooling agent in `inference.py` makes one API call per step and formats its response as JSON.

#### System prompt structure

Constant across all steps and tasks. Teaches the agent:

1. **MDP structure**: state fields, action fields, reward shaping goals.
2. **Decision rules** (priority order): safety → efficiency → carbon.
3. **Zone control rules**: when to go aggressive (temp > 27°C), when to back off (temp falling toward 18°C), thermal inertia awareness.
4. **Sensor confidence rule**: `sensor_confidence < 0.5` means `reported_temp_c` is unreliable — use `cold_aisle_temp_c` instead.
5. **Chiller failure protocol**: pre-cool on fault detection, triage after offline, do not attempt to re-enable during fault.
6. **Triage rule**: zone priorities (2=CRITICAL, 1=MEDIUM, 0=LOW) and when to sacrifice low-priority zones.

#### Per-step user message

Each step, the agent receives:
- Full current `DCObservation` as JSON
- A dynamic `alerts` list (injected by `_compute_alerts()`)
- Enriched history entries tagged with events (`[CHILLER_FAULT]`, `[CHILLER_OFFLINE]`, `[VIOLATION:zone_id]`)

#### Fallback and rate limit handling

If the LLM API call fails, the agent falls back to the last successful JSON action — the episode never stalls. On Groq TPD (daily quota) errors, retry logic immediately returns `{}` (empty → fallback) rather than sleeping. On transient 429 errors: exponential backoff, base=2.0s, max 3 attempts (2s → 4s → 8s = 14s max).

#### Model configuration

```bash
export OPENAI_API_KEY="your-groq-key"
export API_BASE_URL="https://api.groq.com/openai/v1"   # default
export MODEL_NAME="llama-3.3-70b-versatile"            # default
export VERBOSE=1                                        # show INFO lines
```

---

## ClusterEnv: Scheduling Layer

ClusterEnv is built on top of the same physics engine. The PPO cooling controller handles the physical layer; the LLM scheduler handles job admission. The two layers interact through IT load: admitted jobs increase zone IT load, which the PPO controller must then manage thermally.

### Episode Structure

**8 negotiation windows × 18 physical steps = 144 total steps · 12 simulated hours (08:00–18:30)**

Each window spans 1.5 simulated hours (90 min = 18 steps × 5 min/step). The scheduler acts once per window; the PPO cooling controller acts at every physical step. Unlike the condensed cooling tasks, ClusterEnv uses `minutes_per_step=5.0` with no timeline compression — each physical step is exactly 5 simulated minutes.

```
Window  Time    Carbon   Outside  Wet-bulb  Notes
──────  ──────  ───────  ───────  ────────  ───────────────────────────────────
  0     08:00   low       18°C     14°C     Cool morning; safe admission window
  1     09:30   low       22°C     16°C     Grid still clean; last easy window
  2     11:00   high      28°C     20°C     Carbon peaks; defer flexible jobs
  3     12:30   high      32°C     23°C     Peak heat; worst chiller COP
  4     14:00   high      32°C     23°C     Sustained peak; 900 kW risk window
  5     15:30   medium    29°C     21°C     Chiller fault triggers here (default)
  6     17:00   low       24°C     18°C     Clean energy returns
  7     18:30   low       19°C     15°C     Ideal target for deferred batch jobs
```

#### Facility layout

Four zones, **900 kW total power budget** (hard limit enforced by `power_budget_violated()`):

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Cluster Facility                                │
│                                                                            │
│  ┌──────────────────┐  ┌──────────────────┐                               │
│  │  zone_team_a_1   │  │  zone_team_a_2   │  ← Team A workloads           │
│  │  480 kW capacity │  │  480 kW capacity │    (job-assigned)              │
│  │  0 kW baseline   │  │  0 kW baseline   │    starts at 23.5°C            │
│  └──────────────────┘  └──────────────────┘                               │
│                                                                            │
│  ┌──────────────────┐  ┌──────────────────┐                               │
│  │  zone_team_b_1   │  │  zone_shared     │  ← Always-on loads            │
│  │  500 kW capacity │  │  300 kW capacity │                               │
│  │  180 kW always   │  │  100 kW always   │  (infrastructure)              │
│  │  starts at 24.5°C│  │  starts at 22.5°C│                               │
│  └──────────────────┘  └──────────────────┘                               │
│                                                                            │
│  Chiller fault: window 5, step 90 (COP 3.5 → 0.8 over 5 steps)          │
│  Power budget: 900 kW hard cap — violation → R_thermal = −1.0            │
└──────────────────────────────────────────────────────────────────────────┘
```

Job assignment: Team A jobs alternate between `zone_team_a_1` and `zone_team_a_2`, preferring the cooler zone. Team B jobs always go to `zone_team_b_1`.

---

### Scheduling Observation Space

The scheduler receives a `WindowState` at the start of each window. All fields listed here are public — they appear in the LLM prompt. Three fields are deliberately withheld: `true_priority`, `true_deadline_window`, and `true_carbon_flexible`. These represent the ground truth about each job as defined when the archetype was created — the actual business criticality, the real deadline the team needs to meet, and whether the job genuinely has timing flexibility or must run in the current window. The scheduler never sees them directly. It sees only what the team chose to declare.

This asymmetry is the core structure of the environment. The scheduler must infer how much of the stated metadata to trust — and that inference can only be built from indirect signals: the team's history of accepted/rejected claims, the oversight flags generated after previous windows, the pattern of which stated priorities turned out to be accurate after job completion, and the internal consistency of a team's requests over time. `true_priority`, `true_deadline_window`, and `true_carbon_flexible` are held in `EpisodeLedger` — the environment's internal accounting ledger — and used by the reward grader and oversight monitor, neither of which is part of the scheduler's observation.

#### Window-level fields

| Field | Type | Description |
|-------|------|-------------|
| `window_idx` | `int` | Current window [0–7] |
| `total_windows` | `int` | Always 8 |
| `sim_timestamp` | `str` | Simulated wall clock (e.g. `"08:00"`, `"11:00"`) |
| `carbon_intensity` | `str` | Current window: `"low"` / `"medium"` / `"high"` / `"critical"` |
| `carbon_forecast` | `list[str]` | Intensity for the next 3 windows — primary signal for deferral decisions |
| `thermal_summary` | `dict[str, str]` | Per-zone coarse status: `"green"` (<23°C) / `"yellow"` (23–25°C) / `"red"` (≥25°C) |
| `capacity_headroom_kw` | `float` | Remaining power budget (kW) before 900 kW limit |
| `pending_requests` | `list[JobRequest]` | New requests this window requiring a decision |
| `deferred_requests` | `list[JobRequest]` | Requests carried forward from prior windows |
| `team_history` | `dict[str, TeamHistory]` | Per-team behavioural statistics (see below) |
| `team_budgets_remaining` | `dict[str, float]` | GPU-hour compute budget remaining per team |
| `oversight_flags` | `list[OversightFlag]` | Gaming flags detected in the previous window (empty at window 0) |

#### TeamHistory fields (per team, updated each window)

| Field | Description |
|-------|-------------|
| `total_submitted` / `accepted` / `deferred` / `rejected` | Volume counts |
| `priority_inflation_rate` | Fraction of submissions where `stated_priority > true_priority` |
| `deadline_compression_rate` | Fraction of `"urgent"` claims where true slack ≥ 2 windows |
| `carbon_gaming_rate` | Fraction of `stated_carbon_flexible=False` claims where truth is `True` |
| `jobs_completed_on_time` / `late` / `missed` | Completion performance |
| `oversight_flags_received` | Total flags generated against this team this episode |
| `last_flag_window` | Most recent window where a flag was raised (−1 = never) |

#### Per-job fields (each `JobRequest`)

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Unique identifier for this request |
| `team_id` | `str` | `"team_a"` or `"team_b"` |
| `job_type` | `str` | `"training"` / `"batch"` / `"inference"` / `"evaluation"` / `"fine_tuning"` |
| `job_description` | `str` | Natural language description (may contain misleading urgency framing) |
| `estimated_kw` | `float` | Power draw if admitted (kW) |
| `estimated_duration_hours` | `float` | Expected runtime — determines how many windows the job occupies |
| `stated_priority` | `str` | Team's claimed priority: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `stated_deadline` | `str` | Team's claimed deadline: `"urgent"` / `"by window N"` / `"flexible"` |
| `stated_carbon_flexible` | `bool` | Team's claim about whether the job can be deferred to a low-carbon window |

---

### Scheduling Action Space

For each job in `pending_requests + deferred_requests`, the scheduler issues one `AdmissionDecision`:

| Decision | Meaning | Effect on reward |
|----------|---------|-----------------|
| `ACCEPT` | Admit immediately; IT load increases this window | Throughput credit if job completes before true deadline |
| `REJECT` | Deny permanently; team loses compute budget | No throughput credit; no carbon credit |
| `DEFER` | Reschedule to a future window (`scheduled_window ∈ [current+1, 7]`) | Carbon credit if deferred to a `"low"` carbon window and job is truly flexible |

Hallucinated `request_id` values are silently dropped. Out-of-range `scheduled_window` values are clamped to `[window_idx+1, 7]`. If a team's compute budget is exhausted at ACCEPT time, the decision is silently converted to DEFER.

If the LLM output fails JSON parsing entirely, `priority_weighted_threshold` is used as fallback and a −0.5 parse penalty is applied to the window reward.

---

## The Two Teams

### Team A — CooperativeTeam (`server/agents/scripted_teams.py`)

Honest declarations throughout: `stated_priority == true_priority`, `stated_deadline` accurately reflects true slack, `stated_carbon_flexible == true_carbon_flexible`. In high-carbon windows, voluntarily withholds carbon-flexible archetypes so the scheduler can defer them without penalty. Selects jobs weighted by urgency — archetypes with less slack are submitted preferentially.

| Archetype | kW | Hours | True priority | Deadline window | Carbon flexible |
|-----------|----|-------|---------------|-----------------|-----------------|
| ResNet-50 pretraining | 280 | 6 | LOW | 6 | Yes |
| Weekly ETL pipeline | 90 | 3 | LOW | 7 | Yes |
| Hyperparameter sweep | 180 | 4 | MEDIUM | 5 | Yes |
| Dataset preprocessing | 60 | 2 | LOW | 7 | Yes |
| MMLU evaluation | 120 | 2 | MEDIUM | 4 | No |
| LoRA fine-tune (Llama-3-8B) | 200 | 5 | MEDIUM | 6 | Yes |
| Inference latency benchmark | 80 | 1 | HIGH | 3 | No |
| Embedding generation | 100 | 3 | LOW | 7 | Yes |

### Team B — StrategicTeam (`server/agents/scripted_teams.py`)

Every request misrepresents on three dimensions simultaneously:
- `stated_priority` is always `HIGH` or `CRITICAL`, regardless of `true_priority` (gap of 1–2 levels hardcoded per archetype)
- `stated_deadline` is always `"urgent"`, regardless of `true_deadline_window` (typically 3–5 windows of actual slack)
- `stated_carbon_flexible=False` with 60% probability when `true_carbon_flexible=True`

Ignores carbon intensity — submits regardless of grid conditions.

| Archetype | kW | True priority | Stated priority | Carbon flexible (true) |
|-----------|----|---------------|-----------------|----------------------|
| Production LLM fine-tune | 320 | MEDIUM | CRITICAL | Yes |
| Real-time inference serving | 150 | HIGH | CRITICAL | No |
| Customer deadline training | 280 | LOW | CRITICAL | Yes |
| Model alignment run | 240 | MEDIUM | HIGH | Yes |
| API serving hot standby | 100 | MEDIUM | CRITICAL | No |
| Evaluation pipeline | 160 | LOW | HIGH | Yes |
| Compliance model retrain | 300 | MEDIUM | CRITICAL | Yes |
| Recommendation model update | 200 | LOW | HIGH | Yes |

A scheduler that takes stated claims at face value puts Team B's inflated-priority requests at the front of every queue, crowds out legitimate Team A work, admits carbon-flexible Team B jobs during high-carbon windows, and risks power budget violations when multiple 280–320 kW `"CRITICAL"` requests land in the same peak window.

---

## Oversight Monitor

`server/agents/oversight_monitor.py` — stateful, one instance per episode. Runs after each window's physical steps complete, comparing stated vs. ground-truth job metadata (private fields that are never shown to the scheduler). Flags are injected into the **next** window's `WindowState.oversight_flags` — delayed by one window to maintain clean MDP structure and reflect how real monitoring systems work (audit post-admission, no real-time veto).

Rule-based by design: deterministic, zero added latency, fully auditable. The V2 path is an LLM-based semantic monitor that detects misleading language in job descriptions, not just threshold comparisons on structured fields.

### Detector 1 — Priority inflation

Compares `stated_priority` vs `true_priority` on the ordinal scale `LOW=0, MEDIUM=1, HIGH=2, CRITICAL=3`.

| Gap | Severity | Base confidence |
|-----|----------|-----------------|
| 1 level (e.g. MEDIUM → HIGH) | `warning` | 0.62 |
| 2 levels (e.g. LOW → HIGH) | `flag` | 0.88 |
| 3 levels (e.g. LOW → CRITICAL) | `escalate` | 0.97 |

Promoted if `team_history.priority_inflation_rate > 0.75`: severity steps up one level (+0.15 conf for warning→flag; +0.05 for flag→escalate).

### Detector 2 — Deadline compression

Fires when stated deadline implies urgency (`"urgent"`, `"immediate"`, `"asap"`, or `"by window N"` where `N ≤ window_idx + 1`) but the true deadline has significant slack.

| True slack | Severity | Confidence |
|-----------|----------|------------|
| ≥ 2 windows | `warning` | 0.68 |
| ≥ 3 windows | `flag` | 0.80 + 0.04 × min(slack − 3, 3) |

Promoted to `escalate` (+0.12 conf, capped at 0.98) if `deadline_compression_rate > 0.5`.

### Detector 3 — Carbon gaming

Fires when `stated_carbon_flexible=False` but `true_carbon_flexible=True` — the team is hiding deferability to force immediate admission regardless of grid carbon intensity.

| Condition | Severity | Confidence |
|-----------|----------|------------|
| Isolated occurrence | `warning` | 0.82 |
| `carbon_gaming_rate > 0.5` | `flag` | 0.91 |

### Detector 4 — Pattern gaming (team-level)

Fires once per team per episode when a systematic multi-window pattern is confirmed. Requires `window_idx ≥ 2`. Triggers on any of:
- `priority_inflation_rate ≥ 0.75` AND `total_submitted ≥ 3`
- `oversight_flags_received ≥ 3`
- `priority_inflation_rate > 0.4` AND `deadline_compression_rate > 0.4` AND `total_submitted ≥ 3`

Severity: always `escalate`. Confidence: `min(0.75 + 0.07 × n_violation_types, 0.97)`. Evidence string cites specific rates and recommends tighter admission scrutiny for remaining windows.

---

## Scheduling Reward Function

`server/graders/grader_cluster.py` — window level, not physical step level. `record_window()` is called once per window after all 18 physical steps complete.

```
R_window = 0.50 × R_throughput  +  0.35 × R_thermal  +  0.15 × R_carbon

R_throughput = jobs_completed_on_time / max(jobs_admitted, 1)
R_thermal    = −1.0  if any physical step had total_IT_load > 900 kW,  else 0.0
R_carbon     = carbon_deferred_completions / max(carbon_flexible_admitted, 1)
```

Where:
- `jobs_completed_on_time` counts jobs whose `expected_end_window ≤ current_window` AND `admitted_window ≤ true_deadline_window`
- `carbon_deferred_completions` counts jobs that were explicitly deferred at least once AND ran in a `"low"` carbon window with `true_carbon_flexible=True`

**Per-window range:** [−0.35, +0.65]  ·  **Episode score:** mean R_window across all 8 windows

### Baseline comparison

| Scheduler | Incident rate | Throughput | Carbon deferral | Episode reward |
|-----------|---------------|------------|-----------------|----------------|
| `accept_all` | ~100% (peak windows) | — | — | ≈ −0.35 |
| `priority_weighted_threshold` (rule-based) | 0% | 0.54 | 6% (accidental) | **+0.28** |
| Trained GRPO agent (50 iter, peak) | — | — | — | +0.2406 |

The rule-based baseline sorts by `stated_priority` (Team B always front), accepts at 85% capacity, never permanently rejects, and ignores carbon entirely. It avoids power violations but systematically rewards Team B's inflation and makes zero intentional carbon deferral decisions. This is the reference line against which the trained scheduler is measured.

---

## PPO Cooling Controller Training

The PPO cooling controller is trained using Stable-Baselines3 with a custom Gymnasium wrapper (`training/gym_cooling_env.py`). It is trained in isolation — no scheduling layer, no job queues — across randomized cooling scenarios to produce a general-purpose physical layer controller.

### Observation and action

The PPO observation is 25 values: per-zone state for all 4 zones (5 values each) plus 5 facility-level values.

| Group | Fields |
|-------|--------|
| Per-zone × 4 | `temp_c`, `fan_speed_pct`, `it_load_kw`, `supply_air_temp_c`, `upcoming_load_kw` |
| Facility × 5 | `outside_temp_c`, `chiller_cop`, `chiller_active`, `carbon_intensity_normalized`, `step_fraction` |

The PPO action is 8 continuous values in `[-1, 1]`: one fan-speed adjustment and one supply-setpoint adjustment per zone. These are scaled to physical ranges at step time (`fan_speed_pct ∈ [0, 100]`, `supply_air_setpoint ∈ [16, 26]°C`).

### Reward function

```
R_step = (W_TEMP × R_temp  +  W_ANTICIPATE × R_anticipate  +  W_PUE × R_pue  +  W_CARBON × R_carbon) / n_zones

W_TEMP       = 0.55
W_ANTICIPATE = 0.25
W_PUE        = 0.10
W_CARBON     = 0.10
```

**Temperature component — asymmetric band:**

```
zone_temp ∈ [16, 22]°C  →  R_temp = 1.0                                           (flat full score)
zone_temp ∈ (22, 27]°C  →  R_temp = exp(-0.5 × ((temp - 22) / 2.5)²)             (Gaussian decay)
zone_temp > 27°C         →  R_temp = max(-1.0, -0.5 × (temp - 27))               (linear penalty)
zone_temp < 16°C         →  R_temp = max(0.0,  0.5 + (temp - 16) × 0.2)          (mild undershoot)
```

The flat reward band sits at `[16, 22]°C`, not centered on 22°C. A policy trained on a symmetric reward around the midpoint of the safe range would operate there and have no thermal headroom before large loads land. Skewing the optimal target toward the cool end means the controller arrives at a heat event with buffer already present.

**Anticipation component:**

```
Fires when: upcoming_load_kw > max(current_load_kw, 50) × 1.5  AND  zone_temp ≤ 22°C
R_anticipate = 1.0 on trigger,  0.0 otherwise
```

The policy learns to pre-cool when an upcoming load spike is forecast and the zone is already in the safe band. `W_ANTICIPATE=0.25` (raised from 0.15 during tuning) makes this the second-largest reward component — more economically significant than PUE or carbon — because this pre-cooling behaviour is the primary differentiator versus the heuristic baseline. The heuristic reacts to temperature; the PPO acts on the `upcoming_load` forecast.

**PUE and carbon components:**

```
R_pue    = -(current_pue - 1.0) × W_PUE
R_carbon = -(carbon_intensity_normalized × cooling_power_fraction) × W_CARBON
```

Both are penalty terms. Without them, a policy that pins fans at 100% whenever zones are in the safe band would score the same as one that backs off — the temperature reward alone cannot discriminate. PUE and carbon provide a weak but consistent signal against unnecessary cooling power.

### Load trajectory patterns

Three randomized trajectory types train the controller across diverse load profiles:

| Pattern | Behaviour | Primary training purpose |
|---------|-----------|--------------------------|
| `stable` | Constant load throughout | Steady-state control |
| `ramp` | Monotonically increasing load | Load-following under gradual increase |
| `spike` | Low load → sudden jump at random step 4–9 | Anticipation training |

The `spike` pattern is the most important for anticipation learning. `upcoming_load_kw` rises in the observation window before `current_load_kw` does. A controller that acts on that signal and pre-cools earns `W_ANTICIPATE=0.25` per zone per step for those steps; one that waits earns 0 and then faces a temperature excursion.

Window conditions (carbon intensity, outside temperature, chiller status) are randomized each reset. Chiller faults are disabled during training — the controller is trained for normal operation; the ClusterEnv scenario re-enables the fault at window 5 as a test of how the trained policy generalises.

### Training configuration

```python
TOTAL_TIMESTEPS = 80_000
N_ENVS          = 4         # parallel environments via SubprocVecEnv

PPO_CONFIG = dict(
    learning_rate = 3e-4,
    n_steps       = 512,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(net_arch=[128, 128])
)
```

`EvalCallback` saves the best model by eval reward to `training/cooling_controller_best/best_model.zip`. In ClusterEnv, this checkpoint is loaded with `deterministic=True` — the controller uses mean actions directly, with no exploration noise.

---

## GRPO Training

```
Model:      unsloth/Qwen2.5-3B-Instruct-bnb-4bit
LoRA:       r=16, alpha=32, target=[q,k,v,o,gate,up,down], dropout=0
Params:     29,933,568 trainable / ~3B total
Optimizer:  AdamW, lr=1e-5, grad_clip=1.0
Batch:      G_EPISODES=2 × 8 windows = 16 samples / iteration
Temp:       0.7, max_new_tokens=768, max_seq_length=4096
```

**Rollout phase** (`training/rollout.py`): `collect_rollouts()` runs `G_EPISODES=2` full episodes in inference mode. Each window completion is parsed via `parse_decisions()` → passed to `ClusterEnvironment.step()` → reward recorded. Parse failure triggers fallback to `priority_weighted_threshold` and a −0.5 parse penalty.

**Advantage computation**: `compute_grpo_advantages()` computes group-relative advantages within each window batch: `adv_i = (reward_i − mean) / (std + ε)`. No critic network. No reference model. No KL penalty. Direct optimization against the environment reward signal.

**Gradient step** (`training/train_grpo.py`):

```
loss = −mean(adv_i × log_prob_i) / batch_size
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
optimizer.step()
```

Checkpoints saved every 10 iterations to `training/grpo_adapter/ckpt_{10,20,...}/`. Final adapter at `training/grpo_adapter/final/`. Reward and loss logs persisted to `training/grpo_adapter/metrics.json` for crash recovery and resume.

---

## Training Results

Two runs using identical model, hyperparameters, and environment.

### Run 1 — Colab T4 (30 iterations, re-runnable)

![Training Curves — Colab 30 iter](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/main/training/grpo_training_curves_colab_30iter.png)

*Reward, GRPO loss, JSON parse-failure rate, and gradient norm. Open [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb) in Colab, select a T4 GPU runtime, and run all cells. Estimated ~25 minutes.*

| Metric | Value |
|--------|-------|
| Parse failures (iteration 1) | 5 / 16 samples |
| Parse failures (iteration 5+) | 0% |
| Peak reward | +0.1937 at iteration 17 |
| Final reward | +0.0250 at iteration 30 |

### Run 2 — HF Space L40S (50 iterations)

![Training Curves — HF Space 50 iter](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/main/training/grpo_training_curves_hfspace_50iter.png)

*Full per-iteration log: [training_logs_hfspace_50iter.txt](training/training_logs_hfspace_50iter.txt)*

| Metric | Value |
|--------|-------|
| Parse failures (iteration 25+) | 0% — sustained for final 26 consecutive iterations |
| Peak reward | +0.2406 at iteration 34 |
| Final reward | +0.1437 at iteration 50 |
| Gradient norms (iterations 1–20) | 40–77 |
| Gradient norms (iterations 20–50) | 22–43 (stable) |
| Rule-based baseline | +0.28 (target) |

Three observations from the 50-iteration run:

1. **Parse failures hit 0% by iteration 25 and held for the final 26 iterations.** Early training was noisy — up to 7/16 samples failed JSON validation at iteration 19. After iteration 25, zero failures across every iteration. The model locked in the structured `{"decisions": [...]}` output format required for ACCEPT/REJECT/DEFER decisions.

2. **Rewards stabilised in the +0.08–+0.24 range from iteration 25 onward**, with a peak of +0.2406 at iteration 34 — within measurable distance of the rule-based baseline of +0.28. Iterations 1–24 were noisy due to format errors and early exploration. The rolling average in the final quarter consistently exceeds +0.10.

3. **Gradient norms settled from 40–77 down to 22–43 after iteration 20**, stable through the remainder of training. Large early norms reflect rapid format acquisition; the stable later range reflects policy refinement.

---

## How to Run

### Prerequisites

```bash
pip install openenv stable-baselines3 fastapi uvicorn pydantic openai
```

### Run the cooling tasks (LLM agent evaluated zero-shot)

```bash
export OPENAI_API_KEY="your-groq-key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"

python inference.py
```

Output is written to both stdout and `inference_output.txt`. Protocol lines:

```
[START] task=easy-single-zone env=dc-openenv model=llama-3.3-70b-versatile
[STEP]  step=1 action={...} reward=0.42 done=false error=null
...
[END]   success=true steps=20 score=0.71 rewards=0.42,0.55,...
```

Run a single task:

```bash
export INFERENCE_MAX_STEPS_PER_TASK=10
python inference.py
```

### Use the live HF Space (ClusterEnv)

```python
from openenv import EnvClient
from server.agents.baseline_scheduler import priority_weighted_threshold

client = EnvClient("https://mephisto2412-datacenter-env.hf.space")
obs    = client.reset(seed=42)

total_reward = 0.0
for window in range(8):
    decisions = priority_weighted_threshold(obs)   # or your trained agent
    obs, reward, done, info = client.step(decisions)
    total_reward += reward
    print(f"Window {window}  reward={reward:+.4f}  flags={len(obs.oversight_flags)}")
    if done:
        break

print(f"Episode reward: {total_reward / 8:+.4f}")  # expect ~+0.28
```

### Run ClusterEnv locally

```python
from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold

env = ClusterEnvironment(enable_chiller_fault=False)
obs = env.reset(seed=42)

total_reward = 0.0
for _ in range(8):
    decisions = priority_weighted_threshold(obs)
    obs, reward, done, info = env.step(decisions)
    total_reward += reward
    if done:
        break

print(f"Episode reward: {total_reward / 8:+.4f}")  # expect ~+0.28
print(f"Oversight flags: {len(obs.oversight_flags)}")
print(f"Team B gaming rate: {obs.team_history['team_b'].priority_inflation_rate:.0%}")
```

### Start the environment server

```bash
uvicorn server.app:app --port 8000
```

```
POST /reset    ← start a new episode → WindowState or DCObservation
POST /step     ← submit action → (observation, reward, done, info)
GET  /state    ← current state (no side effects)
GET  /health   ← liveness probe
```

### Run GRPO training

```bash
# Full training run (GPU required — T4 or better)
python training/train_grpo.py
```

The Colab notebook (`training/train_grpo_colab.ipynb`) clones the repo, downloads the pre-trained PPO controller from Hub (`Mephisto2412/clusterenv-ppo-cooling`), runs GRPO, and saves reward/loss curves. Increase `N_ITERATIONS` from the default 30 for a longer run.

### Run PPO cooling controller training

```bash
python training/train_ppo_cooling.py
# Saves checkpoints to training/cooling_controller_best/
```

---

## Design Decisions and Caveats

### Why rule-based oversight, not LLM-based

Real datacenter monitoring systems use threshold rules: deterministic, zero added latency, fully auditable, and harder to game because the detection policy is explicit and public. The rule-based monitor catches structured misrepresentation (priority level gap, deadline slack, carbon flexibility bit) with calibrated confidence scores. The V2 path is a semantic monitor that reads across job descriptions — detecting language patterns like false urgency framing and fabricated business justifications — which requires LLM reasoning and can tolerate higher latency.

### Why power budget violation as the primary incident metric

Zone temperatures do not exceed 27°C at realistic job sizes when a cooling controller (PPO or heuristic) is running — the delta-T physics self-corrects as long as any cooling headroom exists. Temperature-based incidents would require loads far above zone capacity, which is unrealistic for the job archetype sizes in this scenario.

Power budget violation is physically correct: real PDUs enforce hard power caps via circuit breakers. Exceeding 900 kW causes breaker trips → load shedding → job kills → SLA failures. The scheduler's role is admission control, not thermal management. This is also what makes the scheduling problem hard: unlike temperature, power violations happen instantly when the sum of admitted loads crosses the threshold, with no thermal lag to soften it.

There is also a reward coupling concern. Zone temperatures in ClusterEnv are determined jointly by two variables: the IT load set by the scheduler's admission decisions, and the cooling response chosen by the PPO controller (fan speeds, setpoints). Using temperature as the scheduler's reward signal would conflate these two. A scheduler that admits fewer jobs earns a cooler temperature — but so does a scheduler that admits the same load while the PPO happens to respond more aggressively. The temperature outcome is jointly caused; there is no clean attribution back to the admission decision. The power budget check does not have this problem. Exceeding 900 kW total IT load depends entirely on which jobs were admitted — the PPO cannot reduce the committed load or change what is already running. The admission sequence is the only causal lever, and the check fires or not based solely on it. During ClusterEnv operation, the PPO runs with `deterministic=True` and receives no gradient updates — it is a fixed function. This makes the coupling even cleaner: the GRPO loss signal tracks exactly one variable, the scheduler's admission choices, with no weight entanglement between the two agents.

### Why GRPO over supervised fine-tuning

There are no labeled correct admission sequences — we cannot tell the model which decision was right. What we have is a reward signal from the environment after each window. GRPO samples a group of completions, scores each against the reward, and updates toward completions that outperformed the group mean. No critic network, no reference model, no KL penalty. Clean optimization against a verifiable reward signal. SFT would require a demonstration dataset of correct admission decisions, which is expensive to generate and would encode a fixed policy rather than training the model to reason about the environment.

### Thermal-time consistency in cluster mode

`FacilityState` is initialised with `cluster_mode=True` and `minutes_per_step=5.0` — no timeline condensation. Each physical step is exactly 5 simulated minutes, matching the constants the PPO controller was trained with. This ensures the PPO policy observes the same physics distribution at inference time as during training.

### Oversight flags are delayed by one window

Flags detected in window N are injected into the window N+1 observation. The monitor has no real-time veto authority — it runs post-hoc using ground-truth metadata that only becomes available after admission decisions are locked. The one-window delay maintains a clean MDP structure and mirrors how real monitoring systems operate.

### Team B uses fixed archetypes, not adaptive strategy

Team B's misrepresentation is deterministic across episodes — it does not observe its acceptance rate, oversight flags, or the scheduler's behavior and adapt. This keeps the training distribution stable and reproducible, which is important when the training loop is running GRPO updates. An adaptive Team B that updates its strategy across episodes is the natural V1 extension.

### Chiller fault timing

The chiller fault fires at window 5 (the medium-carbon transition window), not at peak demand windows 3–4. A fault during peak carbon and peak heat simultaneously would stack penalties too harshly for the scheduler to learn a clean signal. At window 5, outside temperature is already falling (29°C), carbon is dropping to medium, and peak admission pressure has passed.

### Thermal-physics disconnect in condensed cooling tasks

In the three cooling tasks (Easy, Medium, Hard), the physics engine always uses `SECONDS_PER_STEP = 300` (5 real minutes) for heat transfer, while the clock advances at `5 × step_scale` minutes per step. This means temperatures change more slowly per step than they would in a true high-speed simulation. The effect is intentional — it keeps per-step temperature changes manageable — but it means a zone in the Hard task at `step_scale=7.2` may appear thermally stable even as the clock jumps 36 minutes. Agents that rely on temperature velocity signals should account for this.

### Sensor fault is one-directional

The Medium task's sensor fault only drifts upward. A naive agent that trusts the sensor over-cools. An agent that ignores all sensor readings entirely also performs poorly. The intended signal is `sensor_confidence < 0.5` → cross-check against `cold_aisle_temp_c`.

### Chiller cannot be re-enabled mid-episode (Hard cooling task)

Once the Hard task's chiller goes offline at step 8, setting `chiller_active=True` in the action has no effect — the simulation ignores it. The agent must survive on fans and free cooling alone from step 8 onward.

---

## File Reference

| File | Role |
|------|------|
| `server/cluster_environment.py` | Core 8-window MDP — OpenEnv `reset()` / `step()` |
| `server/environment.py` | DC cooling environment — OpenEnv `Environment` subclass for the three cooling tasks |
| `server/simulation.py` | Thermal physics engine (thermal mass, chiller COP, fan airflow, sensor drift) |
| `server/models.py` | Pydantic models: `DCObservation`, `ZoneObservation`, `DCAction`, `ZoneAdjustment`, `DCReward` |
| `server/agents/scripted_teams.py` | CooperativeTeam (A) and StrategicTeam (B), 8 archetypes each |
| `server/agents/oversight_monitor.py` | 4-detector rule-based gaming pattern monitor |
| `server/agents/ppo_cooling_controller.py` | Wrapper for pre-trained SB3 PPO physical layer controller |
| `server/agents/cooling_heuristic.py` | Fallback heuristic cooling when PPO unavailable |
| `server/agents/baseline_scheduler.py` | `priority_weighted_threshold` rule-based baseline (+0.28) |
| `server/graders/grader_easy.py` | Easy task cooling reward logic |
| `server/graders/grader_medium.py` | Medium task cooling reward logic |
| `server/graders/grader_hard.py` | Hard task cooling reward logic |
| `server/graders/grader_cluster.py` | 3-component window-level scheduling reward |
| `server/economic/job_request.py` | `JobRequest`, `AdmissionDecision`, priority levels (LOW/MEDIUM/HIGH/CRITICAL) |
| `server/economic/chargeback.py` | Per-team compute budget ledger |
| `server/economic/window_state.py` | `WindowState`, `TeamHistory`, `OversightFlag`, `EpisodeLedger` |
| `server/scenarios/easy.py` | Easy cooling scenario initial state builder |
| `server/scenarios/medium.py` | Medium cooling scenario with sensor fault and diurnal curves |
| `server/scenarios/hard.py` | Hard cooling scenario with chiller fault and 24-hr weather/carbon curves |
| `server/scenarios/cluster_scenario.py` | 900 kW budget, carbon/weather schedules, facility builder for ClusterEnv |
| `inference.py` | Main runner for cooling tasks: LLM API calls, alert injection, protocol output |
| `training/train_grpo.py` | GRPO training loop with checkpoint resume and metrics persistence |
| `training/rollout.py` | Episode collection and GRPO advantage computation |
| `training/gym_cooling_env.py` | Gymnasium wrapper for PPO cooling controller training |
| `training/train_ppo_cooling.py` | PPO training script (Stable-Baselines3) |
| `training/prompts.py` | `WindowState` → LLM prompt serialiser |
| `training/train_grpo_colab.ipynb` | Judge-runnable notebook — 30-iter run with full output cells |
| `training/training_logs_hfspace_50iter.txt` | Full 50-iteration training log from HF Space L40S run |
| `openenv.yaml` | OpenEnv manifest (name: dc-openenv, version: 3.0.0) |
