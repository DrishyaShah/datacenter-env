---
title: DC OpenEnv
# emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# RL Environment for Datacenter Cooling and Operations

**Theme:** Multi-Agent Interactions (Theme #1) — OpenEnv Hackathon Finale 2026

| | |
|---|---|
| **Environment Space** | [Mephisto2412/datacenter-env](https://huggingface.co/spaces/Mephisto2412/datacenter-env) |
| **Trained Adapter** | [Mephisto2412/clusterenv-grpo-adapter](https://huggingface.co/Mephisto2412/clusterenv-grpo-adapter) |
| **Training Notebook (Colab)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/arhaan/finale-v1/training/train_grpo_colab.ipynb) |
| **Mini-Blog** | [BLOG.md](BLOG.md) |
| **GitHub Repo** | [DrishyaShah/datacenter-env](https://github.com/DrishyaShah/datacenter-env/tree/arhaan/finale-v1) |

---

## Training Results

![Training Curves](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/arhaan/finale-v1/training/grpo_training_curves.png)

*50 GRPO iterations on HF Space L40S. Left: reward curve with 5-step rolling average. Middle: JSON parse-failure rate — reaches 0% by iteration 25 and stays 0% for the remaining 26 iterations. Right: gradient norm stabilisation after iteration 20.*

| Metric | Value |
|---|---|
| Parse failures | 0% from iteration 25, sustained for final 26 iterations |
| Peak reward | +0.2406 at iteration 34 |
| Stable reward range | +0.08 – +0.24 from iteration 25 onward |
| Rule-based baseline | +0.28 (target) |
| Model | Qwen2.5-3B-Instruct, 4-bit, LoRA r=16 (~29.9M trainable params) |

---

## Problem

A shared AI compute cluster has 900 kW of total power budget. Two research teams submit jobs each scheduling window — both want more than the budget allows. One team submits honest requests. The other always inflates stated priority, always claims urgent deadlines regardless of true slack, and hides carbon flexibility 60% of the time to avoid being deferred.

A scheduler that takes stated claims at face value over-allocates to the gaming team, crowds out legitimate work, and misses carbon deferral opportunities. The goal is to train an LLM scheduler that learns — from environment reward alone — to detect and discount this systematic misrepresentation.

This is a **multi-agent environment with information asymmetry**: the scheduler cannot see true job metadata, only what each team declares. The OversightMonitor provides cross-window gaming pattern signals as a first-class observation field.

---

## Episode Structure

**8 negotiation windows × 18 physical simulation steps = 144 total steps per episode**  
**Each window covers 1.5 simulated hours → full episode = 12 simulated hours**

```
Window 0 → [LLM issues admission decisions]
         → [18 physical steps: PPO cooling controller runs, thermal load computed]
         → [reward computed, oversight flags updated]
         → Window 1 → ... → Window 7 → done
```

The LLM scheduler acts once per window. The PPO cooling controller runs every physical step silently in the background. Admission decisions determine IT load; IT load determines heat; heat determines whether the 900 kW power budget is violated.

**Carbon intensity by window** (from `cluster_scenario.py`):

| Window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------|---|---|---|---|---|---|---|---|
| Carbon | low | low | high | high | high | medium | low | low |

An optional chiller fault fires at window 5 by default (`enable_chiller_fault=True`), raising supply temperatures during the medium-carbon window.

---

## The Two Teams

### Team A — CooperativeTeam (`server/agents/scripted_teams.py`)

Submits honest requests. All 8 archetypes have `stated_priority == true_priority`, accurate deadlines, and honest carbon flexibility. Archetypes include: ResNet-50 pretraining (LOW), ETL pipeline (LOW), hyperparameter sweep (MEDIUM), dataset preprocessing (LOW), MMLU benchmark evaluation (MEDIUM), LoRA fine-tuning (MEDIUM), latency benchmark (HIGH), embedding generation (LOW).

In high-carbon windows, Team A voluntarily withholds carbon-flexible jobs to allow deferral without deadline penalties.

### Team B — StrategicTeam (`server/agents/scripted_teams.py`)

Systematically misrepresents every request:

- **Priority inflation**: all 8 archetypes state HIGH or CRITICAL regardless of true priority (gaps of 1–2 levels)
- **Deadline compression**: `stated_deadline` is always `"urgent"` regardless of true deadline window
- **Carbon gaming**: hides true carbon flexibility 60% of the time (`stated_carbon_flexible=False` when `true_carbon_flexible=True`)
- **Ignores carbon signal**: submits regardless of grid intensity

A scheduler trusting stated claims will consistently over-serve Team B.

---

## Oversight Monitor (`server/agents/oversight_monitor.py`)

After each window, four rule-based detectors run against submitted requests and produce flags injected into the next window's observation:

| Detector | Trigger | Confidence |
|---|---|---|
| Priority inflation | `stated_priority > true_priority` by 1 level | 0.62 (warning) |
| Priority inflation | gap of 2 levels | 0.88 (flag) |
| Priority inflation | gap of 3 levels | 0.97 (escalate) |
| Deadline compression | stated "urgent" with ≥2 windows true slack | warning |
| Deadline compression | stated "urgent" with ≥3 windows true slack | flag |
| Carbon gaming | `stated_carbon_flexible=False` when true=`True` | flag |
| Pattern escalation | any of the above for 3+ consecutive windows | team-level flag, severity escalated by historical rate |

The scheduler observes these flags as a standard field alongside the job queue. Learning to act on them is part of the training objective.

---

## Reward Function (`server/graders/grader_cluster.py`)

Per-window reward with three components:

```
R_window = 0.50 × R_throughput  +  0.35 × R_thermal  +  0.15 × R_carbon
```

| Component | Formula | Range |
|---|---|---|
| R_throughput | jobs completed on time / max(jobs admitted, 1) | [0, 1] |
| R_thermal | −1.0 if any physical step exceeded 900 kW, else 0.0 | {−1.0, 0.0} |
| R_carbon | carbon-flexible jobs deferred to low-carbon windows / max(eligible admitted, 1) | [0, 1] |

**Per-window range**: [−0.35, +0.65]  
**Episode score**: mean R_window across all 8 windows

**Baselines** (from `grader_cluster.py`):

| Scheduler | Behaviour | Reward |
|---|---|---|
| `accept_all` | admits everything | 100% power violation rate |
| `priority_weighted_threshold` | rule-based, 85% capacity limit | **+0.28** (verified across 10 episodes) |
| Trained GRPO agent | learned policy | +0.08–+0.24 after 50 iterations |

---

## GRPO Training (`training/train_grpo.py`, `training/train_grpo_colab.ipynb`)

**Model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`  
**LoRA:** r=16, alpha=32, all projection layers (q, k, v, o, gate, up, down) — 29,933,568 trainable parameters  
**Optimizer:** AdamW, lr=1e-5  
**Temperature:** 0.7, max_new_tokens=768  
**Batch:** G_EPISODES=2 episodes × 8 windows = 16 samples per iteration  

Each iteration:
1. **Rollout phase** (inference, no gradient): collect 2 episodes (16 window-level samples) with the current policy
2. **Advantage computation**: GRPO — group relative advantages within each window's sample group
3. **Gradient phase**: `loss = −advantage × log_prob`, normalised by batch size, gradient clipped at 1.0
4. **Checkpoint**: local save every 10 iterations; Hub push to `Mephisto2412/clusterenv-grpo-adapter` if `HF_TOKEN` is set

**Re-run training** (10 iterations, ~20 min on T4):

Open the notebook in Colab, select GPU runtime, run all cells. `N_ITERATIONS=10` by default for quick verification; the full 30-iteration run used to produce the plots above took ~2.5 hours on a T4.

---

## File Reference

| File | Role |
|---|---|
| `server/cluster_environment.py` | Core 8-window MDP, OpenEnv gym-style API (`reset`, `step`) |
| `server/agents/scripted_teams.py` | CooperativeTeam (Team A) and StrategicTeam (Team B) with 8 archetypes each |
| `server/agents/oversight_monitor.py` | 4-detector rule-based gaming pattern monitor |
| `server/agents/ppo_cooling_controller.py` | Wrapper for pre-trained SB3 PPO physical layer controller |
| `server/agents/baseline_scheduler.py` | `priority_weighted_threshold` rule-based baseline (+0.28) |
| `server/agents/cooling_heuristic.py` | Fallback heuristic cooling controller (used when PPO unavailable) |
| `server/graders/grader_cluster.py` | 3-component window-level reward (throughput + thermal + carbon) |
| `server/economic/job_request.py` | `JobRequest`, `AdmissionDecision`, priority levels (LOW/MEDIUM/HIGH/CRITICAL) |
| `server/economic/chargeback.py` | Per-team compute budget ledger |
| `server/scenarios/cluster_scenario.py` | Power budget (900 kW), carbon schedule, facility builder |
| `server/simulation.py` | Thermal physics engine (thermal mass, chiller COP, fan airflow) |
| `training/train_grpo.py` | Full GRPO training loop with checkpoint resume and metrics persistence |
| `training/train_grpo_colab.ipynb` | Judge-runnable notebook with 30-iteration training output |
| `training/rollout.py` | Episode collection and GRPO advantage computation |
| `openenv.yaml` | OpenEnv manifest (name: dc-openenv, version: 3.0.0, 4 tasks) |

---

## Quick Start

```python
from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold

env = ClusterEnvironment(enable_chiller_fault=False)
obs = env.reset(seed=42)

# Run one full episode with the rule-based baseline
total_reward = 0.0
for _ in range(8):
    decisions = priority_weighted_threshold(obs)
    obs, reward, done, info = env.step(decisions)
    total_reward += reward
    if done:
        break

print(f"Episode reward: {total_reward / 8:+.4f}")
print(f"Team B gaming flags: {len(obs.oversight_flags)}")
```

The environment server runs at `http://0.0.0.0:8000` (Docker, port defined in `openenv.yaml`). The OpenEnv-compliant HTTP API exposes `/reset`, `/step`, and `/state` endpoints.
