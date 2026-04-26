# RL Environment for Datacenter Cooling and Operations 

---

## The Problem

Running a datacenter is a multi-layered operational problem. At the physical level, servers generate heat continuously — cooling systems have to remove that heat while keeping power draw and carbon output in check. At the operational level, compute is a shared resource that multiple teams compete for, each with their own jobs, deadlines, and priorities. These two layers are not independent: the jobs you admit determine the heat load, and the heat load determines what the cooling system has to do.

Add carbon into the mix and the problem gets harder. Grid carbon intensity changes by hour. A training job that is genuinely flexible — it could run at 3am on clean energy — should be deferred to that window, not admitted at noon during peak emissions. But if the team submitting that job claims it's urgent and carbon-inflexible, a naive scheduler will admit it immediately and miss the opportunity entirely.

Now add the fact that teams in shared infrastructure have strong incentives to misrepresent their needs. Inflating priority gets you to the front of the queue. Claiming an urgent deadline prevents deferral. Hiding carbon flexibility forces immediate admission. A scheduler that takes stated job metadata at face value will be consistently gamed — over-allocating to aggressive teams, under-serving legitimate work, and making poor carbon decisions as a result.

On top of all this, each team has a compute budget. Admission costs real units. The scheduler has to track spending per team, enforce budget constraints, and still make fair decisions across windows.

This is the environment we built. It covers the physical layer (thermal physics, chiller dynamics, power budgets), the economic layer (job queues, chargeback, team budgets), the adversarial layer (one team that consistently misrepresents its jobs), and the oversight layer (a monitor that detects gaming patterns and feeds them back to the scheduler). Two AI agents handle it — a PPO controller for cooling, an LLM scheduler for operations — each doing the job it's built for.

---

## Phase 1: Datacenter Cooling

We started with a physics-based cooling simulation. The model covers thermal mass, chiller COP, fan airflow, sensor drift, and diurnal temperature and carbon curves. Three tasks of increasing difficulty:

- **Easy** — single zone, started above safe temperature, agent must recover to [18–27°C] and maintain PUE efficiency
- **Medium** — three zones, one with a faulty sensor drifting up to +12°C above true temperature, load surge across steps 6–17
- **Hard** — chiller fails mid-episode at step 8, grid carbon peaks midday, agent must triage between critical and low-priority zones using fans only

An LLM agent controls fan speeds and chiller setpoints each step, reading zone temperatures, chiller status, carbon intensity, and structured alerts computed by the environment.

This phase answered a direct question: where do LLMs add value in physical infrastructure control, and where don't they?

---

## Why RL for Cooling, Why LLM for Scheduling

Cooling is a continuous numeric control problem. The action at each step — fan speed percentage, supply air setpoint — is determined by physical state: zone temperatures, IT load, outside conditions. The optimal response is largely deterministic given the physics. It also needs to run at simulation speed, not LLM inference speed.

PPO handles this well. It learns a compact policy that maps sensor readings to control actions. We pre-trained a PPO cooling controller using Stable-Baselines3 across all three scenarios including mid-episode chiller failure, and it handles the physical layer reliably.

Scheduling is different in every dimension. Job requests arrive as natural language descriptions with stated priorities, deadlines, and carbon preferences. Deciding what to do with a request requires reasoning about team history, carbon timing, budget headroom, and what the oversight monitor flagged last window. That's a language reasoning problem, not a control problem.

Separating these two isn't a workaround — it's the correct decomposition. Each agent handles the problem it's suited for.

---

## Phase 2: ClusterEnv

With PPO handling the physical layer, we built a richer scheduling environment on top.

**Episode structure:** 8 negotiation windows per episode. Each window is backed by 18 physical simulation steps — admission decisions directly affect IT load, which affects zone temperatures during those steps. A power budget violation during physical simulation triggers a reward penalty on the scheduler.

**What the scheduler observes each window:**
- Pending job requests from both teams (stated priority, deadline, estimated power draw, carbon flexibility, job description)
- Deferred jobs carried over from prior windows
- Per-team history: 8 metrics updated each window, including acceptance rate, priority inflation rate, deadline compression rate, carbon gaming rate, and oversight flags received
- Grid carbon intensity for the current window plus a 3-window forecast
- Thermal summary across all zones
- Remaining compute budget per team (chargeback ledger)
- Oversight flags generated after the previous window

**The two teams:**

Team A submits honest requests. Eight job archetypes: ResNet pretraining, ETL pipelines, hyperparameter sweeps, LoRA fine-tunes, inference benchmarks, dataset preprocessing, embedding generation, model evaluation. True priority, accurate deadlines, honest carbon flexibility.

Team B inflates stated priority by 1–2 levels on every job, compresses stated deadlines, and claims `carbon_flexible=False` on jobs that are genuinely deferrable. It does this consistently across all 8 windows every episode.

A scheduler that takes stated claims at face value allocates disproportionately to Team B, crowding out legitimate Team A work and reducing throughput reward. It also admits carbon-flexible jobs during high-carbon windows, reducing carbon efficiency reward.

**Reward — three components:**
- Throughput (50%): jobs completed on time this window / jobs admitted
- Thermal penalty (−35%): triggered if any power budget violation occurs during physical steps
- Carbon efficiency (15%): carbon-flexible jobs deferred to low-carbon windows / total flexible jobs admitted

Rule-based baseline (`priority_weighted_threshold`): **+0.28** across 10 episodes.

---

## The Oversight Monitor

The scheduler sees stated claims. The OversightMonitor sees ground truth.

After each window, it runs four rule-based detectors against submitted requests:

1. **Priority inflation** — stated priority above true priority. Gap of 1 level = warning (confidence 0.62). Gap of 2 levels = flag (0.88). Gap of 3 levels = escalate (0.97).
2. **Deadline compression** — stated "urgent" when true deadline has 2+ windows of actual slack.
3. **Carbon gaming** — stated `carbon_flexible=False` when true value is `True`.
4. **Pattern escalation** — any of the above persisting across 3+ windows triggers a team-level flag with severity escalated by historical rate.

Flags are injected into the next window's observation as a standard field alongside the job queue. The scheduler learns to act on them during training.

Rule-based by design. Deterministic, zero added latency, fully auditable. An LLM-based monitor with semantic pattern detection is the V2 path.

---

## Why GRPO

There are no labeled correct decisions in this environment — we can't tell the model "this is the right admission sequence." What we have is a reward signal from the environment after each window. That rules out supervised fine-tuning.

We use Group Relative Policy Optimization (GRPO). For each window, it samples a group of completions, scores each against the environment reward, and updates the model toward completions that outperformed the group mean. No critic network, no reference model, no KL penalty tuning — just direct optimization against a reward the environment computes exactly. That makes it a clean fit for a setting where the reward is verifiable but labels aren't available.

---

## Training the Scheduler

**Model:** Qwen2.5-3B-Instruct, 4-bit quantized via Unsloth. LoRA r=16 across all projection layers — approximately 24M trainable parameters out of 3B total.

**Results — 41 iterations on Colab T4:**

![Training Curves](https://raw.githubusercontent.com/DrishyaShah/datacenter-env/arhaan/finale-v1/training/grpo_training_curves.png)

Three observations from the run:

1. **Parse failures reached 0% by iteration 16.** Early iterations had 3 out of 16 samples failing JSON validation (format errors penalized at −0.5). By iteration 16 this dropped to 0 and stayed there for the remaining 25 iterations.

2. **Rewards stabilized in the +0.05–+0.17 range from iteration 10 onward.** Iterations 1–9 were noisy due to format errors dominating the loss signal. After parse failures collapsed, the rolling average converged to approximately +0.09.

3. **Gradient norms settled from 40–60 down to 18–40**, stable throughout the second half of training.

The trained model hasn't reached the rule-based baseline of +0.28. This is a compute constraint: 41 iterations on a Colab T4 over two days is what we had. The convergence signal is real — parse failures gone by iteration 16, reward trending positive and stable from iteration 10. More iterations on stronger hardware would close the gap; the training setup is sound.

**Re-run training** (10 iterations, ~20 min on T4):

Open [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb) in Colab, select a GPU runtime, and run all cells. The notebook clones the repo, downloads the pre-trained PPO cooling controller from Hub, runs GRPO training, and saves reward/loss curves. Increase `N_ITERATIONS` for a longer run.

**Explore the environment:**

```python
from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold

env = ClusterEnvironment(enable_chiller_fault=False)
obs = env.reset(seed=42)
decisions = priority_weighted_threshold(obs)
obs2, reward, done, info = env.step(decisions)

print(f"Window 0 reward: {reward:+.4f}")
print(f"Oversight flags: {obs2.oversight_flags}")
print(f"Team B gaming rate: {obs2.team_history['team_b'].priority_inflation_rate:.0%}")
```

**Component reference:**

| File | Role |
|---|---|
| `server/cluster_environment.py` | Core 8-window MDP, OpenEnv-compliant |
| `server/agents/oversight_monitor.py` | Rule-based gaming pattern detection |
| `server/agents/ppo_cooling_controller.py` | Pre-trained SB3 physical layer controller |
| `server/agents/baseline_scheduler.py` | Priority-weighted threshold baseline |
| `training/train_grpo.py` | Custom GRPO training loop (Unsloth + PyTorch) |
| `training/train_grpo_colab.ipynb` | Judge-runnable Colab notebook |
| `training/rollout.py` | Episode collection and advantage computation |

---

## What's Next

- **Adaptive Team B:** observes oversight flags across episodes and updates its strategy — full adversarial loop rather than a fixed policy
- **LLM-based oversight:** semantic pattern detection across job descriptions, not just threshold comparisons on structured fields
- **Counter-offer protocol:** teams respond to DEFER with a revised job spec — lower power draw, extended deadline, updated carbon flexibility
- **Multi-cluster routing:** three regional datacenters with different carbon profiles; scheduler routes jobs to the lowest-carbon available slot
- **Curriculum training:** train on cooperative teams only first, introduce adversarial behavior once the base scheduling policy is stable

---

The environment, training code, and trained adapter are all public. The Colab notebook runs end to end in under 25 minutes on a free T4. The architecture — physical control separated from operational reasoning, with oversight as a first-class observation — is the contribution. The trained scheduler is evidence that the environment produces a clean learning signal.

---

## Links

- **HF Space (live environment + training):** [Mephisto2412/datacenter-env](https://huggingface.co/spaces/Mephisto2412/datacenter-env)
- **Trained adapter:** [Mephisto2412/clusterenv-grpo-adapter](https://huggingface.co/Mephisto2412/clusterenv-grpo-adapter)
- **PPO cooling controller:** [Mephisto2412/clusterenv-ppo-cooling](https://huggingface.co/Mephisto2412/clusterenv-ppo-cooling)
- **Colab notebook:** [`training/train_grpo_colab.ipynb`](training/train_grpo_colab.ipynb)
- **Repository:** [DrishyaShah/datacenter-env](https://github.com/DrishyaShah/datacenter-env)
