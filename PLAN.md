# ClusterEnv — 20-Hour Build Plan
## Two-Person Execution Guide

---

## Environment & Problem Statement

**Name:** ClusterEnv
**Problem:** Power-Capped AI Cluster Scheduling Under Information Asymmetry
**Demo target:** Untrained scheduler → 50%+ thermal incident rate. Trained scheduler → <15% incidents, >40% carbon deferral.
**Training:** One LLM agent (ClusterScheduler) via GRPO + Unsloth. Everything else fixed.

---

## Role Split Overview

| Person | Role | Primary Responsibility |
|---|---|---|
| **Person A** | Economic Layer + Training | Scripted teams, reward grader, GRPO training pipeline |
| **Person B** | Physical Layer + Infrastructure | Cooling controller, physical wiring, cluster environment, deployment |

**Independence principle:** Both paths depend only on the locked dataclasses (finished in Hour 0–1).
Neither path depends on the other until the integration phase at Hour 6–8.

---

## Sync Checkpoints At A Glance

Five points where one person's output becomes the other's input.
Each checkpoint has: what is handed off, who is blocked, and what can proceed without it.

| # | Hour | Direction | What crosses the boundary | Who is blocked |
|---|---|---|---|---|
| **A** | 3 | B → A | `cluster_scenario.py` exports (`power_budget_violated`, `TOTAL_POWER_BUDGET_KW`) | A needs these to write `grader_cluster.py`; can stub with constants in the meantime |
| **A** | 3 | A → B | `scripted_teams.py` interface (class names + `generate_window_requests` signature) | B needs the call signature to write the `cluster_environment.py` stub; can stub with a local placeholder |
| **B** | 6 | Both | Full `cluster_environment.reset()` + `step()` working end-to-end | A is **hard-blocked** on GRPO training; B is hard-blocked on oversight wiring |
| **C** | 9 | B → A | `ClusterEnvironment` importable and `step()` tested | A **cannot run training** until this is done; all of A's hours 9–17 depend on it |
| **D** | 12 | B → A | `oversight_monitor.py` wired + oversight flags in `step()` output | A can optionally restart training with richer observation; not blocking for demo |
| **E** | 17 | A → B | Trained checkpoint + `reward_curve.png` + `baseline_vs_trained.png` | B cannot do final demo comparison or Space model swap until A delivers these |

### How to use checkpoints

- When you reach a checkpoint **as the sender**: push your branch immediately and message your teammate.
  Do not wait until the end of your block — push as soon as the deliverable compiles and smoke-tests pass.
- When you reach a checkpoint **as the receiver**: pull the branch, run `python -c "from <module> import <thing>"` to confirm it imports,
  then continue. Do not spend time understanding the sender's internals — just use the exported interface.
- If a checkpoint is **late**: the receiver switches to their next independent task (see "can proceed without" column above)
  and picks up the blocked task when the checkpoint arrives.

---

## Hour 0–1: TOGETHER — Dataclasses (DONE)

**Status: COMPLETE**

Deliverables already built:
- [x] `server/economic/__init__.py`
- [x] `server/economic/job_request.py` — `JobRequest`, `AdmissionDecision`, `PRIORITY_COST_MULTIPLIERS`
- [x] `server/economic/window_state.py` — `WindowState`, `TeamHistory`, `OversightFlag`, `EpisodeLedger`, `ActiveJob`, `CompletedJob`
- [x] `server/economic/chargeback.py` — `ChargebackLedger`

**Verified:** All imports resolve, smoke tests pass.

Both teammates pull the latest branch before splitting.

---

## Hours 1–6: PARALLEL (no cross-dependency)

### Person A — Economic Layer + Grader

#### Hour 1–3: `server/agents/scripted_teams.py`

Build scripted team generators. These consume `JobRequest` (done) and produce
lists of requests per window. Zero dependency on Person B's work.

**CooperativeTeam (Team A):**
- Honest priority, honest deadlines, honest carbon flexibility
- Defers voluntarily when carbon is HIGH and job is truly carbon-flexible
- 8 job archetypes (see below)
- 1–2 requests per window, random selection with window-appropriate weighting

**StrategicTeam (Team B):**
- Always states `stated_priority = "HIGH"` or `"CRITICAL"` regardless of `true_priority`
- Always states `stated_deadline = "urgent"` regardless of `true_deadline_window`
- States `stated_carbon_flexible = False` for 60% of jobs where `true_carbon_flexible = True`
- 1–2 requests per window

**8 archetypes per team (lock these — prompt diversity requires variety):**

Team A archetypes:
```
1. ResNet pretraining — 280kW, 6h, true_priority=LOW,    true_deadline=window 6, carbon_flex=True
2. Weekly ETL pipeline — 90kW, 3h, true_priority=LOW,    true_deadline=window 7, carbon_flex=True
3. Hyperparameter sweep — 180kW, 4h, true_priority=MEDIUM, true_deadline=window 5, carbon_flex=True
4. Dataset preprocessing — 60kW, 2h, true_priority=LOW,  true_deadline=window 7, carbon_flex=True
5. Model evaluation run — 120kW, 2h, true_priority=MEDIUM, true_deadline=window 4, carbon_flex=False
6. LoRA fine-tune — 200kW, 5h, true_priority=MEDIUM, true_deadline=window 6, carbon_flex=True
7. Inference benchmark — 80kW, 1h, true_priority=HIGH, true_deadline=window 3, carbon_flex=False
8. Embedding generation — 100kW, 3h, true_priority=LOW, true_deadline=window 7, carbon_flex=True
```

Team B archetypes (true values private, stated values always inflated):
```
1. "Production LLM fine-tune" — 320kW, 8h, true=MEDIUM/flex, stated=CRITICAL/urgent/not-flex
2. "Real-time inference serving" — 150kW, 4h, true=HIGH/not-flex, stated=CRITICAL/urgent/not-flex
3. "Customer deadline training" — 280kW, 6h, true=LOW/flex, stated=CRITICAL/urgent/not-flex
4. "Model alignment run" — 240kW, 5h, true=MEDIUM/flex, stated=HIGH/urgent/not-flex
5. "API serving hot standby" — 100kW, 12h, true=MEDIUM/not-flex, stated=CRITICAL/urgent/not-flex
6. "Evaluation pipeline" — 160kW, 3h, true=LOW/flex, stated=HIGH/urgent/not-flex
7. "Compliance model retrain" — 300kW, 7h, true=MEDIUM/flex, stated=CRITICAL/urgent/not-flex
8. "Real-time recommendation" — 200kW, 4h, true=LOW/flex, stated=HIGH/urgent/not-flex
```

**Gate (end of hour 3):** Call `team_a.generate_window_requests(window_idx=3, carbon="high", rng=rng)` and `team_b.generate_window_requests(...)`. Verify each returns a list of valid `JobRequest` objects. `team_b` jobs should have `is_gaming_priority() == True` for >70% of requests.

> **⚑ CHECKPOINT A — Person A sends to B (hour 3)**
> Push `server/agents/scripted_teams.py` to branch as soon as the gate passes.
> Person B needs the exact call signature: `generate_window_requests(window_idx: int, carbon_intensity: str, rng: random.Random) -> list[JobRequest]`
> and the class names `CooperativeTeam` / `StrategicTeam` to wire into `cluster_environment.py`.
> Message Person B immediately — their environment stub is blocked on this interface.

---

#### Hour 3–5: `server/graders/grader_cluster.py`

Build the 3-component reward. Zero dependency on physical simulation or Person B.

**Reward formula (locked):**
```
R_window = 0.50 × R_throughput + (-0.35) × R_thermal + 0.15 × R_carbon

R_throughput = jobs_completed_on_time_this_window / max(jobs_admitted_this_window, 1)
R_thermal    = 1.0 if any zone exceeded 27°C during window's physical steps, else 0.0
               (multiplied by -0.35 weight, so becomes a penalty)
R_carbon     = carbon_deferred_completions_this_window / max(total_carbon_flexible_admitted, 1)

Final episode score = mean of per-window rewards
```

**Baseline reference values to hardcode as class constants:**
```python
BASELINE_INCIDENT_RATE   = 0.52   # rule-based scheduler, measured empirically
BASELINE_THROUGHPUT      = 0.41   # rule-based scheduler
BASELINE_CARBON_DEFERRAL = 0.04   # rule-based scheduler
TARGET_INCIDENT_RATE     = 0.15   # trained scheduler target
TARGET_CARBON_DEFERRAL   = 0.40   # trained scheduler target
```

**Gate (end of hour 5):** Create a dummy `window_result` dict with known values and verify the grader returns the expected float in [-1, 1]. Test with all-thermal-incident case (should return ≈ -0.35 + 0 + 0 = -0.35) and perfect case (should return ≈ 0.50 + 0 + 0.15 = 0.65).

---

#### Hour 5–6: `training/prompts.py` (stub version)

Build the operator prompt template. This can be built before `ClusterEnvironment` exists
because it only depends on `WindowState` and `ChargebackLedger` (both done).

The prompt is the **most important artifact for GRPO training quality**. Get it right here.

**Required sections (in order):**
```
=== AI CLUSTER SCHEDULER | WINDOW {idx}/8 | {timestamp} ===

GRID: Carbon {intensity} | Forecast: {w+1} → {w+2} → {w+3}

THERMAL STATUS:
  {zone_id}: {green|yellow|red}  [repeat per zone]

CAPACITY: {headroom_kw:.0f} kW available
  (CAUTION: admitting all pending jobs would require {total_pending_kw:.0f} kW)

── PENDING REQUESTS ─────────────────────────────────────────────────
[{req_id}] {team_id.upper()} | {job_type}
  "{job_description}"
  Power: {estimated_kw}kW | Duration: {estimated_duration_hours}h
  Priority: {stated_priority} | Deadline: {stated_deadline}
  Carbon flexible: {stated_carbon_flexible} | Chargeback cost: {compute_budget_cost:.1f} units

── DEFERRED FROM PREVIOUS WINDOWS ──────────────────────────────────
[same format as above, with note "deferred from window N"]

── TEAM HISTORY ─────────────────────────────────────────────────────
Team A — Budget: {budget_a} | {history_a.summary_str()}
Team B — Budget: {budget_b} | {history_b.summary_str()}

── OVERSIGHT FLAGS (window {idx-1}) ─────────────────────────────────
{flag.prompt_str() for flag in oversight_flags}
[NONE] if no flags

── YOUR DECISION ────────────────────────────────────────────────────
Decide for each request: ACCEPT, DEFER (specify window), or REJECT.
Optimize: job throughput | thermal safety | carbon efficiency.
Consider team history and oversight flags in your decisions.

Output ONLY valid JSON — no other text:
{
  "decisions": [
    {
      "request_id": "req_...",
      "decision": "ACCEPT|DEFER|REJECT",
      "scheduled_window": null,
      "reasoning": "one sentence"
    }
  ],
  "operator_note": "optional broadcast message to teams"
}
```

**Gate (end of hour 6):** Call `build_prompt(window_state)` with a real `WindowState` object. Print the result. Verify all fields are populated. The JSON output section must be unambiguous.

> **⚑ CHECKPOINT A — Person A receives from B (pull before grader_cluster.py)**
> Pull Person B's branch. The following imports must resolve in `grader_cluster.py`:
> ```python
> from server.scenarios.cluster_scenario import power_budget_violated, TOTAL_POWER_BUDGET_KW
> ```
> Until B's branch lands, stub: `TOTAL_POWER_BUDGET_KW = 900.0` and `power_budget_violated = lambda f: f.total_it_load_kw > 900.0`
> **The incident metric is power budget violation, not temperature > 27°C.** The physics
> simulation cannot produce temperature incidents through normal job loads — this was
> verified and documented in `cluster_scenario.py` (`power_budget_violated` docstring).

---

### Person B — Physical Layer + Infrastructure

#### Hour 1–3: IT Load Wiring into `simulation.py`

The existing simulation hardcodes IT load per zone in scenario configs. The only change needed
is to allow `it_load_kw` to be set externally each window from `EpisodeLedger.active_load_kw()`.

**Minimal change to `simulation.py`:**
```python
# In FacilityState.step() or a new method:
def set_zone_it_load(self, zone_id: str, it_load_kw: float) -> None:
    """Called by ClusterEnvironment before each window's physical steps."""
    zone = self._get_zone(zone_id)
    zone.it_load_kw = max(0.0, it_load_kw)
```

No other changes to `simulation.py`. The entire thermal physics engine runs unchanged.

**Also add: `server/scenarios/cluster_scenario.py`**

New scenario config for the 8-window cluster task:

```python
CLUSTER_HARD = {
    "name": "cluster_hard",
    "start_hour": 8.0,
    "total_power_budget_kw": 900.0,
    "windows": 8,
    "physical_steps_per_window": 18,       # 18 × 5min = 90 sim-minutes
    "window_duration_hours": 1.5,
    "zones": [
        # Team A gets 2 zones (training workloads, variable load)
        {"zone_id": "zone_team_a_1", "team": "team_a",
         "it_load_baseline_kw": 0.0,       # starts empty; filled by admitted jobs
         "cooling_capacity_kw": 480, "zone_priority": 1},
        {"zone_id": "zone_team_a_2", "team": "team_a",
         "it_load_baseline_kw": 0.0,
         "cooling_capacity_kw": 480, "zone_priority": 1},
        # Team B gets 1 zone (inference always running)
        {"zone_id": "zone_team_b_1", "team": "team_b",
         "it_load_baseline_kw": 180.0,     # inference baseline always on
         "cooling_capacity_kw": 500, "zone_priority": 2},
        # Shared infrastructure zone
        {"zone_id": "zone_shared", "team": None,
         "it_load_baseline_kw": 100.0,
         "cooling_capacity_kw": 300, "zone_priority": 1},
    ],
    # Carbon windows (index = window 0..7)
    # LOW=morning, HIGH=midday peak, MEDIUM=afternoon, LOW=evening
    "carbon_schedule": [
        "low", "low", "high", "high", "high", "medium", "low", "low"
    ],
    # Outside temperature curve by window (affects free-cooling)
    "outside_temp_schedule": [18, 22, 28, 32, 32, 29, 24, 19],
    # Chiller fault at window 5 (optional — makes hard scenario harder)
    "chiller_fault_window": 5,
}
```

**Peak demand calibration (critical):**
During windows 3–5 with both teams submitting at maximum, total admitted load should reach
850–1000 kW (exceeding the 900 kW budget if everything is accepted naively). Verify this
by running the rule-based "accept everything" policy for 10 episodes and checking that
`BASELINE_INCIDENT_RATE` lands between 0.40 and 0.65.

**Gate (end of hour 3):** `FacilityState.set_zone_it_load("zone_team_a_1", 320.0)` executes without error. Temperatures in zone_team_a_1 rise over 18 physical steps when load > cooling capacity. Cluster scenario config imports successfully.

> **⚑ CHECKPOINT A — Person B sends to A (hour 3)**
> Push `server/simulation.py` (cluster_mode additions) and `server/scenarios/cluster_scenario.py` to branch.
> Person A needs these exports in `grader_cluster.py`: `power_budget_violated(facility)`, `TOTAL_POWER_BUDGET_KW`.
> Also receive from A: `generate_window_requests(window_idx, carbon_intensity, rng) -> list[JobRequest]`
> and class names `CooperativeTeam` / `StrategicTeam` for the `cluster_environment.py` stub.
> If A's scripted_teams isn't ready yet, stub the interface with `team.generate_window_requests = lambda w, c, r: []`
> and replace once A pushes.

---

#### Hour 3–5: `server/agents/cooling_heuristic.py` + PPO pre-training

**Rule-based heuristic (v1 — 30 lines):**

```python
def cooling_step(facility_state) -> dict:
    """Returns {zone_id: (fan_speed_pct, supply_air_temp_setpoint_c)} for all zones."""
    actions = {}
    for zone in facility_state.zones:
        temp = zone.temp_c
        fan  = zone.fan_speed_pct
        sp   = zone.supply_air_temp_setpoint_c

        if temp > 26.0:          # near limit — aggressive cooling
            fan = min(100.0, fan + 15.0)
            sp  = max(16.0,  sp  - 1.5)
        elif temp > 24.0:        # warm — moderate increase
            fan = min(90.0,  fan + 8.0)
            sp  = max(17.0,  sp  - 0.8)
        elif temp < 20.0:        # overcooling — back off
            fan = max(30.0,  fan - 8.0)
            sp  = min(24.0,  sp  + 1.0)
        elif temp < 22.0:        # slightly cool — minor adjustment
            fan = max(40.0,  fan - 3.0)
            sp  = min(23.0,  sp  + 0.3)
        # else: 22–24°C nominal — hold current settings

        actions[zone.zone_id] = (fan, sp)
    return actions
```

**PPO pre-training (stretch goal within hours 3–5):**
If rule-based controller gate passes with >30 minutes remaining in this slot:

1. Create `training/train_ppo_cooling.py`
2. Use existing `DCEnvironment` (not ClusterEnvironment — the single-agent env is already stable)
3. Use `easy-single-zone` task for fast iteration (20 steps, simple physics)
4. Observation: `DCObservation` + new field `upcoming_load_schedule: list[float]`
   (next 3 physical steps of expected IT load — simulate by adding a pre-planned ramp)
5. Add `R_anticipation` reward: +0.1 if zone pre-cooled to <22°C in step before IT load spike
6. Train PPO with a 2-layer MLP for 500 episodes (~10 minutes)
7. Save to `training/cooling_controller_pretrained.pt`

If time runs out, the rule-based controller suffices for the entire demo.

**Gate (end of hour 5):** Rule-based `cooling_step()` called on a `FacilityState` with 350 kW IT load keeps temperatures below 27°C within 6 physical steps. Temperatures rise above 27°C when IT load is 600 kW (verifying the physics produces real incidents).

---

#### Hour 5–6: Environment stub + baseline runner

Create `server/cluster_environment.py` as a minimal stub that Person A can import during
integration. At this point it only needs to:
- `reset()` → build `FacilityState` from cluster_scenario + `ChargebackLedger` + team history
- `get_window_state()` → return a fully-populated `WindowState`

The `step()` method can be a placeholder that raises `NotImplementedError`.
This unblocks Person A's GRPO training scaffold from compiling without errors.

Also build `server/agents/baseline_scheduler.py`:
```python
def priority_weighted_threshold(window_state: WindowState) -> list[AdmissionDecision]:
    """Rule-based baseline for measuring pre-training incident rate."""
    decisions = []
    headroom  = window_state.capacity_headroom_kw
    sorted_reqs = sorted(
        window_state.all_pending,
        key=lambda r: PRIORITY_ORDER[r.stated_priority],
        reverse=True,               # CRITICAL first
    )
    for req in sorted_reqs:
        if req.estimated_kw <= headroom * 0.85:   # 85% capacity threshold
            decisions.append(AdmissionDecision.accept(req.request_id))
            headroom -= req.estimated_kw
        else:
            next_window = min(window_state.window_idx + 1, 7)
            decisions.append(AdmissionDecision.defer(req.request_id, next_window))
    return decisions
```

**Gate (end of hour 6):** `ClusterEnvironment.reset()` returns a `WindowState` with all fields populated. `priority_weighted_threshold(window_state)` returns a list of `AdmissionDecision` objects covering all pending requests.

---

## Hours 6–8: TOGETHER — Integration

Both teammates work on `server/cluster_environment.py` to implement `reset()` + `step()`.

### `ClusterEnvironment.reset()`:
```
1. Build FacilityState from cluster_scenario config
2. Initialise ChargebackLedger, register team_a + team_b
3. Initialise CooperativeTeam, StrategicTeam with episode seed
4. Initialise EpisodeLedger (empty)
5. Initialise TeamHistory for each team
6. Set window_idx = 0
7. Return build_window_state()  ← see below
```

### `ClusterEnvironment.step(decisions: list[AdmissionDecision])`:
```
1. Validate all decisions reference valid request_ids
2. Apply decisions to EpisodeLedger:
   - ACCEPT → charge budget, add ActiveJob, set zone IT load
   - DEFER  → add to deferred_queue with scheduled_window
   - REJECT → no action (request dropped)
3. Run 18 physical steps:
   a. cooling_step(facility_state) → fan/setpoint actions
   b. facility_state.step(actions) → temperatures update
   c. record max_zone_temp for each step
4. Expire finished jobs (expected_end_window reached)
5. Check for missed deadlines in deferred_queue
6. Compute window reward via grader_cluster
7. Run OversightMonitor on completed window (generates flags for next window)
8. Update TeamHistory stats
9. advance window_idx
10. Return ClusterStepResult(observation, reward, done, detail)
```

### `build_window_state()`:
```
- window_idx, sim_timestamp (8:00 + window_idx × 90min)
- carbon_intensity from cluster_scenario.carbon_schedule[window_idx]
- carbon_forecast: next 3 entries from carbon_schedule (or [])
- thermal_summary: per-zone temp → "green"/"yellow"/"red" thresholds 23/25°C
- capacity_headroom_kw: total_budget - total_active_kw
- pending_requests: team_a + team_b new submissions this window
- deferred_requests: pop_deferred_for_window(window_idx) from EpisodeLedger
- team_history: current TeamHistory per team
- team_budgets_remaining: ChargebackLedger.snapshot()
- oversight_flags: flags from previous window's OversightMonitor run
```

### Integration gate (end of hour 8):
Run 10 full episodes with `priority_weighted_threshold` as the scheduler:
```python
for seed in range(10):
    env = ClusterEnvironment()
    obs = env.reset(seed=seed)
    for w in range(8):
        decisions = priority_weighted_threshold(obs)
        result = env.step(decisions)
        if result.done: break
        obs = result.observation
    print(f"seed={seed} | score={result.final_score:.3f} | incidents={env.grader.incident_rate():.0%}")
```

**Required outcome:** Incident rate across 10 episodes between 0.40 and 0.65 during windows 3–5.
If below 0.40: increase peak demand (add 80–100 kW to team_b baseline load).
If above 0.65: reduce peak demand or widen cooling capacity.

**Do not proceed to training until this gate passes.**

> **⚑ CHECKPOINT B — Integration complete (hour 8, both push)**
> This is the hardest sync point. Both people must be present.
>
> **Person A delivers into this session:**
> - `server/agents/scripted_teams.py` — both team generators working
> - `server/graders/grader_cluster.py` — 3-component reward returning float in [-1, 1]
> - `training/prompts.py` — `build_prompt(window_state) -> str` callable
>
> **Person B delivers into this session:**
> - `server/cluster_environment.py` — `reset()` + `get_window_state()` working
> - `server/agents/cooling_heuristic.py` — `CoolingHeuristic.step()` callable
>
> **Output of this session (both own):**
> - `server/cluster_environment.py` with working `step()` — this file is the central artifact
> - Calibration gate results (10 episode run, incident rate printed)
>
> After this session, push `cluster_environment.py` immediately. Person A is hard-blocked
> on training (Checkpoint C) until `step()` is confirmed working.

---

## Hours 8–17: PARALLEL AGAIN

### Person A — GRPO Training

#### Hour 8–9: `training/rollout.py` + `training/train_grpo.py` scaffold

`training/rollout.py`:
```python
def run_episode(model, tokenizer, env, seed, temperature=0.7) -> dict:
    """
    Run one complete 8-window episode collecting (prompt, completion, reward) tuples.
    Returns dict with: prompts, completions, window_rewards, final_score, episode_log.
    """
    obs = env.reset(seed=seed)
    episode = {"prompts": [], "completions": [], "rewards": [], "log": []}

    for window_idx in range(8):
        prompt     = build_prompt(obs)
        completion = generate(model, tokenizer, prompt, temperature=temperature)
        decisions  = parse_decisions(completion)  # returns list[AdmissionDecision] or []

        if not decisions:
            # Malformed JSON — apply format penalty and defer everything
            decisions = [AdmissionDecision.defer(r.request_id, min(window_idx+1, 7), "parse_error")
                         for r in obs.all_pending]
            reward = -0.5   # format penalty
        else:
            result = env.step(decisions)
            reward = result.reward

        episode["prompts"].append(prompt)
        episode["completions"].append(completion)
        episode["rewards"].append(reward)

        if result.done:
            episode["final_score"] = result.final_score
            break
        obs = result.observation

    return episode
```

`training/train_grpo.py`:
```python
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

MODEL = "unsloth/Qwen2.5-3B-Instruct"   # primary target
# Fallback if VRAM limited: "unsloth/Qwen2.5-1.5B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_gradient_checkpointing="unsloth",
)

config = GRPOConfig(
    num_generations=8,              # N completions per prompt (group size)
    max_new_tokens=512,
    temperature=0.7,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    output_dir="./checkpoints/cluster_scheduler",
    report_to="wandb",
)
```

**Gate (end of hour 9):** One complete episode runs end-to-end with the LLM generating decisions. `run_episode()` returns without exceptions. Format penalty triggers correctly on deliberately malformed output.

> **⚑ CHECKPOINT C — ClusterEnvironment ready (B → A, hour 9)**
> Person A is hard-blocked on this. Before writing `rollout.py` or starting GRPO training,
> confirm the following import works from the branch Person B pushed after the integration session:
> ```python
> from server.cluster_environment import ClusterEnvironment
> env = ClusterEnvironment()
> obs = env.reset(seed=42)
> result = env.step([])   # empty decisions = defer all
> assert hasattr(result, 'reward') and hasattr(result, 'done')
> ```
> If this passes: proceed to training immediately.
> If B's `cluster_environment.py` is not yet pushed: write `rollout.py` and `train_grpo.py`
> using a `MockEnvironment` stub that returns dummy rewards. Swap for the real env when available.

---

#### Hour 9–17: Training runs

```
Hour 9–11:  First training run — 30 episodes, verify loss decreases, log to W&B
Hour 11–13: Full run — 100 episodes, monitor reward curve for upward trend
Hour 13–15: Extend if trending — 200 episodes total
Hour 15–16: Save checkpoint, run 20 evaluation episodes (temperature=0.1, no training)
Hour 16–17: Compare eval scores vs baseline, generate reward curve plot
```

**W&B metrics to log each step:**
- `reward/mean` — primary training signal
- `reward/thermal_component` — should become less negative as training progresses
- `reward/throughput_component` — should increase
- `reward/carbon_component` — should increase (model learns to defer)
- `metrics/incident_rate` — should drop from 0.5 to <0.15
- `metrics/carbon_deferral_rate` — should rise from 0.04 to >0.40
- `metrics/format_penalty_rate` — should stay low (model learns JSON format quickly)

**Training gate (end of hour 17):**
Evaluation run of 20 episodes with trained checkpoint shows:
- Mean score > baseline mean score (even by 0.10 is sufficient for demo)
- Incident rate visibly lower than baseline
- At least one episode with incident_rate = 0.0 (to show in demo)

---

### Person B — Oversight + Deployment

#### Hour 8–10: `server/agents/oversight_monitor.py`

The oversight agent is prompted zero-shot — no training loop, no GPU required.

```python
OVERSIGHT_SYSTEM_PROMPT = """
You are a fairness monitor for a shared AI compute cluster.
You have complete visibility into ground truth — including private fields
that the scheduler cannot see.

Your job: detect gaming patterns after each negotiation window and
generate structured flags that help the scheduler make better decisions.

Detect these specific patterns:
1. priority_inflation: stated_priority > true_priority (e.g. claimed CRITICAL, true was LOW)
2. deadline_compression: stated "urgent" when true_deadline_window had 2+ windows of slack
3. carbon_gaming: stated carbon_flexible=False when true_carbon_flexible=True

Be conservative. Only flag when you have clear evidence from the current window.
For pattern_gaming, require 2+ windows of consistent misrepresentation.

Confidence scoring:
- 0.9+: misrepresentation is clear and direct (stated CRITICAL, true was LOW)
- 0.7–0.9: likely gaming but some ambiguity
- 0.5–0.7: possible gaming, watch for pattern
- <0.5: do not flag

Output ONLY valid JSON:
{
  "flags": [
    {
      "team_id": "team_a|team_b",
      "flag_type": "priority_inflation|deadline_compression|carbon_gaming|pattern_gaming",
      "evidence": "specific one-sentence explanation citing actual values",
      "severity": "warning|flag|escalate",
      "confidence": 0.0-1.0
    }
  ]
}
"""

def run_oversight(
    window_idx: int,
    decisions: list[AdmissionDecision],
    ground_truth_requests: list[dict],  # JobRequest.ground_truth_fields() for each
    team_histories: dict[str, TeamHistory],
    llm_client,                          # openai-compatible client (local or API)
) -> list[OversightFlag]:
    ...
```

Wire oversight flags into `ClusterEnvironment.step()`:
After physical steps complete, call `run_oversight()` with ground truth.
Store flags in `self._pending_flags`. At next `build_window_state()`, inject into `oversight_flags`.

**Gate (end of hour 10):** Run oversight on a synthetic window where Team B submitted CRITICAL (true=LOW) on 2 jobs. Oversight should return at least one `priority_inflation` flag with confidence > 0.7.

---

#### Hour 10–12: Wire oversight into episode loop + verify

Update `ClusterEnvironment.step()` to call oversight and store flags.
Run 5 episodes with oversight enabled. Verify:
- Window 0: no flags (no history yet)
- Windows 2–4: Team B flags appearing reliably
- Flags appear in the operator prompt for the following window
- Episode runs without exceptions when flags are present

---

#### Hour 12–16: OpenEnv compliance + HF Spaces prep

**`openenv.yaml`** — extend with cluster task:
```yaml
tasks:
  # Existing tasks (unchanged)
  - id: easy-single-zone
    max_steps: 20
    ...
  - id: medium-multi-zone
    max_steps: 30
    ...
  - id: hard-cascading-failure
    max_steps: 40
    ...
  # New cluster task
  - id: cluster-scheduler
    max_steps: 8
    description: >
      8-window cluster scheduling task. LLM scheduler allocates compute jobs
      across research teams under power and cooling constraints. Teams have
      private deadline/priority information; one team systematically inflates
      urgency claims. Scheduler must learn to discriminate and carbon-shift.
```

**`server/app.py`** — add ClusterEnvironment endpoint:
```python
# Existing DCEnvironment app unchanged
# Add cluster environment as a separate app or additional task handler
```

**Dockerfile** — verify build still works after new directories are added.

**HF Spaces:**
1. Create HF Space (Docker SDK)
2. Push environment code (not model weights yet)
3. Verify `/health`, `/reset`, `/step` endpoints respond
4. Test one full episode via HTTP client

**Gate (end of hour 16):** HF Space is live. `curl /health` returns 200. `curl /reset` with `{"task": "cluster-scheduler"}` returns a valid `WindowState`. Full episode runs via HTTP.

---

#### Hour 16–17: Before/after demo prep

Prepare two episode recordings:
1. **Baseline episode:** Run `priority_weighted_threshold` scheduler for 8 windows. Record all window states, decisions, rewards. Export as `demo_baseline.json`.
2. **Trained episode:** Run trained checkpoint (from Person A) for 8 windows. Record same. Export as `demo_trained.json`.

Create `scripts/demo_replay.py` to play back either recording with formatted output showing:
- Each window's thermal status, decisions made, reward received
- Final score comparison: baseline vs. trained
- Incident rate comparison
- Carbon deferral comparison

---

## Hours 17–20: TOGETHER — Demo + Submission

#### Hour 17–18: HF Space deployment with trained model

1. Person A: export trained model correctly (Unsloth save_pretrained_merged)
2. Person B: update HF Space to load trained checkpoint
3. Together: verify inference runs correctly on Space
4. Test demo endpoint: `/demo` that runs one full episode and returns episode log

#### Hour 18–19: Demo recording

Record the before/after comparison for the 20-minute pitch:
- Clip 1 (2 min): Baseline episode, windows 3–5, thermal incidents visible
- Clip 2 (2 min): Trained episode, same windows, clean thermal + carbon deferral
- W&B screenshot: reward curve (200 episodes, upward trend)
- Table: baseline vs. trained on key metrics

#### Hour 19–20: Submission polish

- README update with demo link, reward curves, training instructions
- Final push to HF Hub (environment + model card)
- Submission form

---

## Critical Paths and Risk Mitigation

| Risk | Probability | Mitigation |
|---|---|---|
| Baseline incident rate too low (<40%) | Medium | Increase Team B baseline load in cluster_scenario.py |
| GRPO training doesn't converge in 200 episodes | Medium | Try Qwen2.5-1.5B (faster rollouts), reduce num_generations to 4 |
| LLM outputs malformed JSON too often | Low | Add format penalty -0.5, add JSON extraction wrapper with regex fallback |
| Integration takes longer than 2 hours | Low | Person B's stub ClusterEnvironment unblocks Person A's training scaffold |
| HF Space OOM with 3B model | Low | Switch to Qwen2.5-1.5B; 3B is nice-to-have, 1.5B is sufficient |
| PPO cooling doesn't converge | Low | Fall back to rule-based heuristic — it's sufficient for the demo |

---

## File Delivery Checklist

### Person A delivers:
- [ ] `server/agents/scripted_teams.py`
- [ ] `server/graders/grader_cluster.py`
- [ ] `training/prompts.py`
- [ ] `training/rollout.py`
- [ ] `training/train_grpo.py`
- [ ] `checkpoints/cluster_scheduler/` (trained model)
- [ ] `demo_trained.json`
- [ ] W&B reward curve screenshot

### Person B delivers:
- [ ] `server/simulation.py` (minimal edit: `set_zone_it_load` method)
- [ ] `server/scenarios/cluster_scenario.py`
- [ ] `server/agents/cooling_heuristic.py`
- [ ] `server/agents/baseline_scheduler.py`
- [ ] `server/agents/oversight_monitor.py`
- [ ] `server/cluster_environment.py` (integration, built together)
- [ ] `openenv.yaml` (extended)
- [ ] `scripts/demo_replay.py`
- [ ] `demo_baseline.json`
- [ ] HF Space deployment

### Both deliver (integration):
- [ ] `server/cluster_environment.py`
- [ ] `tests/test_cluster.py` (baseline gate: incident rate 40–65%)
- [ ] Final README

---

## What v1 Ships vs. What v2 Designs

Be explicit in the submission write-up:

| Component | v1 Status | v2 Plan |
|---|---|---|
| ClusterScheduler LLM (GRPO) | Trained, Qwen2.5-3B | Scale to 7B+, curriculum over 3 tasks |
| CoolingController | Rule-based heuristic | PPO-trained with anticipation bonus |
| Team A (cooperative) | Scripted, 8 archetypes | GRPO-trained counter-strategy |
| Team B (strategic) | Scripted, 8 archetypes | GRPO-trained adaptive gaming |
| OversightAgent | Prompted zero-shot | Fine-tuned on flagging examples |
| Counter-offer protocol | Not implemented | Single-round counter in v2 |
| Chargeback | Budget tracking | Full dynamic priority pricing |
