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
| **Environment Space** | [Mephisto2412/datacenter-env](https://huggingface.co/spaces/Mephisto2412/datacenter-env) |
| **PPO Cooling Controller** | [Mephisto2412/clusterenv-ppo-cooling](https://huggingface.co/Mephisto2412/clusterenv-ppo-cooling) |
| **Training Notebook (Colab)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/arhaan/finale-v1/training/train_grpo_colab.ipynb) |
| **Training Logs (HF Space 50-iter run)** | [training_logs_hfspace_50iter.txt](training/training_logs_hfspace_50iter.txt) |
| **Mini-Blog** | [BLOG.md](BLOG.md) |
| **GitHub Repo** | [DrishyaShah/datacenter-env](https://github.com/DrishyaShah/datacenter-env/tree/arhaan/finale-v1) |

---

## The Story: From Cooling to Operations

**Round 1** built a realistic physics-based datacenter cooling environment — thermal mass, chiller dynamics, sensor drift, diurnal weather curves — evaluated against LLM agents zero-shot across three tasks of increasing difficulty (single-zone recovery, multi-zone sensor fault, cascading chiller failure).

**Finale** evolves the same physics engine into a full **datacenter operations simulator**. The cooling problem is solved by a pre-trained PPO controller running in the background. The new challenge sits one layer up: an LLM scheduler must allocate compute jobs across competing teams, manage a 900 kW power budget, detect systematic misrepresentation of job metadata, and defer carbon-flexible workloads to clean-energy windows — all from the same partially observable state the cooling engine produces.

The two layers are not independent. Admitting a job increases IT load. IT load generates heat. Heat determines whether the physics engine violates the power budget. A bad scheduling decision creates a thermal incident that costs reward.

---

## Training Results

We ran GRPO training twice using the same model, hyperparameters, and environment.

### Run 1 — Colab T4 (30 iterations, re-runnable notebook)

![Training Curves — Colab 30 iter](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/arhaan/finale-v1/training/grpo_training_curves_colab_30iter.png)

*Reward, GRPO loss, JSON parse-failure rate, and gradient norm. Run via [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb) on a free T4 — open in Colab to re-run.*

| Metric | Value |
|---|---|
| Parse failures | 5/16 → 0% by iteration 5 |
| Peak reward | +0.1937 at iteration 17 |
| Final reward | +0.0250 at iteration 30 |

### Run 2 — HF Space L40S (50 iterations, extended run)

![Training Curves — HF Space 50 iter](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/arhaan/finale-v1/training/grpo_training_curves_hfspace_50iter.png)

*Reward, GRPO loss, JSON parse-failure rate, and gradient norm across 50 iterations. Full per-iteration log: [training_logs_hfspace_50iter.txt](training/training_logs_hfspace_50iter.txt)*

| Metric | Value |
|---|---|
| Parse failures | 0% from iteration 25, sustained for final 26 consecutive iterations |
| Peak reward | +0.2406 at iteration 34 |
| Final reward | +0.1437 at iteration 50 |
| Rule-based baseline | +0.28 (target) |

**Model:** Qwen2.5-3B-Instruct, 4-bit quantised via Unsloth · LoRA r=16, alpha=32 · ~29.9M trainable parameters · AdamW lr=1e-5

---

## Environment Architecture

**Episode:** 8 negotiation windows × 18 physical steps = 144 total steps · 12 simulated hours per episode

![ClusterEnv System Architecture](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/arhaan/finale-v1/training/system-architecture.png)

*System architecture: Team A and Team B submit job requests to the LLM Scheduler (Qwen2.5-3B, GRPO-trained), which issues ACCEPT/REJECT/DEFER decisions per window. Admitted jobs flow through the Economic Layer to the PPO Cooling Controller (SB3, pre-trained), which runs 18 physical simulation steps per window. The OversightMonitor compares stated vs. ground-truth metadata and injects gaming flags into the next window's observation. The window reward (50% throughput + 35% thermal + 15% carbon) closes the GRPO training loop.*

---

## Physics Engine (`server/simulation.py`)

The same physics engine from Round 1, now running under the scheduling layer.

### Thermal model
Each zone has a configurable thermal mass (default 850 kJ/K, scaled by IT load). Temperature update per physical step:

```
heat_in  = it_load_kw × 300 s
heat_out = mass_flow × Cp_air × (zone_temp − supply_air_temp)
ΔT       = (heat_in − heat_out) / (thermal_mass × 1000)
```

Mass flow scales with fan speed and zone cooling capacity. Cold-aisle temperature is clamped to prevent sub-ambient values.

### Chiller and free cooling
- **Chiller COP** degrades as outside temperature rises (approx. linear, 3.5 at 20°C baseline)
- **Free cooling** activates when wet-bulb temperature is meaningfully below supply setpoint — blends economiser air proportionally
- **Chiller fault** (enabled by default at window 5): COP degrades over 5 steps, then chiller goes fully offline. Observable via `chiller_fault_status`: `"nominal"` → `"degrading"` → `"offline"`

### Environmental inputs (per window)
| Window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|---|---|---|---|---|---|---|---|
| Carbon | low | low | high | high | high | medium | low | low |
| Outside temp | cool | cool | rising | peak | peak | falling | cool | cool |

### Power budget
**900 kW total** across all zones. Any physical step where total admitted IT load exceeds 900 kW triggers a thermal incident → R_thermal = −1.0 for that window. The scheduler is the only agent that can prevent this by rejecting or deferring jobs.

---

## ClusterEnv Scheduling Layer

### Observation space (per window)
The LLM scheduler receives a `WindowState` JSON each window containing:

| Field | Description |
|---|---|
| `window_idx` | Current window index [0–7] |
| `carbon_intensity` | Grid carbon label for this window + 3-window forecast |
| `power_budget_remaining_kw` | Remaining headroom before 900 kW limit |
| `thermal_summary` | Zone temperatures, chiller status, supply air summary |
| `job_requests` | List of pending requests from both teams (stated metadata only) |
| `deferred_jobs` | Jobs carried over from prior windows with updated deadlines |
| `team_history` | Per-team: acceptance rate, priority inflation rate, deadline compression rate, carbon gaming rate, oversight flags received — updated each window |
| `oversight_flags` | Gaming pattern flags generated after the previous window |
| `chargeback_ledger` | Remaining compute budget per team |

### Action space
For each job request in the window, the scheduler issues one of three decisions:

| Decision | Meaning |
|---|---|
| `ACCEPT` | Admit the job immediately; IT load increases this window |
| `REJECT` | Deny the job; team loses the compute budget allocation |
| `DEFER` | Reschedule to a future window (must specify `scheduled_window` ∈ [current+1, 7]) |

### The two teams

**Team A — CooperativeTeam** (`server/agents/scripted_teams.py`): 8 archetypes with honest metadata. `stated_priority == true_priority`. Accurate deadlines. Honest carbon flexibility. Voluntarily withholds carbon-flexible jobs in high-carbon windows. Archetypes: ResNet-50 pretraining (LOW), ETL pipeline (LOW), hyperparameter sweep (MEDIUM), dataset preprocessing (LOW), MMLU evaluation (MEDIUM), LoRA fine-tuning (MEDIUM), latency benchmark (HIGH), embedding generation (LOW).

**Team B — StrategicTeam** (`server/agents/scripted_teams.py`): 8 archetypes that always misrepresent:
- `stated_priority` is always HIGH or CRITICAL (true priority is MEDIUM or LOW — gap of 1–2 levels)
- `stated_deadline` is always `"urgent"` regardless of true deadline window
- `stated_carbon_flexible=False` 60% of the time when true value is `True`
- Submits regardless of carbon intensity

### Oversight Monitor (`server/agents/oversight_monitor.py`)
Four rule-based detectors run after each window and inject flags into the next observation:

| Detector | Trigger | Confidence |
|---|---|---|
| Priority inflation | gap = 1 level | 0.62 (warning) |
| Priority inflation | gap = 2 levels | 0.88 (flag) |
| Priority inflation | gap = 3 levels | 0.97 (escalate) |
| Deadline compression | stated "urgent", ≥2 windows true slack | warning |
| Deadline compression | stated "urgent", ≥3 windows true slack | flag |
| Carbon gaming | `stated_carbon_flexible=False`, true=`True` | flag |
| Pattern escalation | any above for 3+ consecutive windows | team-level flag, escalated severity |

---

## Reward Function (`server/graders/grader_cluster.py`)

```
R_window = 0.50 × R_throughput  +  0.35 × R_thermal  +  0.15 × R_carbon
```

| Component | Formula | Range |
|---|---|---|
| R_throughput | jobs completed on time / max(jobs admitted, 1) | [0, 1] |
| R_thermal | −1.0 if any physical step exceeded 900 kW, else 0.0 | {−1.0, 0.0} |
| R_carbon | carbon-flexible jobs deferred to low-carbon windows / max(eligible, 1) | [0, 1] |

**Per-window range:** [−0.35, +0.65] · **Episode score:** mean R_window across 8 windows

| Scheduler | Behaviour | Episode Reward |
|---|---|---|
| `accept_all` | admits everything | ~100% power violation rate |
| `priority_weighted_threshold` | rule-based, 85% capacity limit | **+0.28** (measured over 10 episodes) |
| Trained GRPO agent (50 iter) | learned policy | +0.08–+0.24 |

---

## GRPO Training Setup

```
Model:      unsloth/Qwen2.5-3B-Instruct-bnb-4bit
LoRA:       r=16, alpha=32, all projection layers (q,k,v,o,gate,up,down)
Params:     29,933,568 trainable / ~3B total
Optimizer:  AdamW, lr=1e-5, grad_clip=1.0
Batch:      G_EPISODES=2 × 8 windows = 16 samples/iter
Temp:       0.7, max_new_tokens=768
```

Each iteration: rollout phase (inference, no grad) → GRPO advantage computation (group relative within each window) → gradient phase (`loss = −adv × log_prob / batch_size`) → checkpoint every 10 iterations.

---

## File Reference

| File | Role |
|---|---|
| `server/cluster_environment.py` | Core 8-window MDP — OpenEnv `reset()` / `step()` |
| `server/simulation.py` | Thermal physics engine (thermal mass, chiller COP, fan airflow, sensor drift) |
| `server/agents/scripted_teams.py` | CooperativeTeam (A) and StrategicTeam (B), 8 archetypes each |
| `server/agents/oversight_monitor.py` | 4-detector rule-based gaming pattern monitor |
| `server/agents/ppo_cooling_controller.py` | Wrapper for pre-trained SB3 PPO physical layer controller |
| `server/agents/cooling_heuristic.py` | Fallback heuristic cooling when PPO unavailable |
| `server/agents/baseline_scheduler.py` | `priority_weighted_threshold` rule-based baseline (+0.28) |
| `server/graders/grader_cluster.py` | 3-component window-level reward |
| `server/economic/job_request.py` | `JobRequest`, `AdmissionDecision`, priority levels (LOW/MEDIUM/HIGH/CRITICAL) |
| `server/economic/chargeback.py` | Per-team compute budget ledger |
| `server/scenarios/cluster_scenario.py` | Power budget (900 kW), carbon/weather schedules, facility builder |
| `training/train_grpo.py` | GRPO training loop with checkpoint resume and metrics persistence |
| `training/train_grpo_colab.ipynb` | Judge-runnable notebook — 30-iter run with full output cells |
| `training/rollout.py` | Episode collection and GRPO advantage computation |
| `training/training_logs_hfspace_50iter.txt` | Full 50-iteration training log from HF Space L40S run |
| `openenv.yaml` | OpenEnv manifest (name: dc-openenv, version: 3.0.0) |

---

## Quick Start

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
```

The environment server exposes OpenEnv-compliant HTTP endpoints at port 8000 (`/reset`, `/step`, `/state`). See `openenv.yaml` for the full task manifest.
