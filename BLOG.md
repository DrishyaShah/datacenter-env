# ClusterEnv: Teaching an LLM to Schedule Under Deception

*A post-training environment where an AI scheduler learns to allocate compute fairly while one team systematically lies about its needs.*

---

## The Problem

Imagine you manage compute allocation for an AI cluster shared between two research teams. Every scheduling window, both teams submit job requests with stated priorities, deadlines, and carbon preferences. But one team has learned to game the system — inflating priorities, compressing deadlines, and misrepresenting flexibility to extract more resources.

You have 900 kW of power budget. Both teams want more than that. Who gets access, and who gets deferred?

This is **ClusterEnv**: an 8-window Markov Decision Process where an LLM scheduler must allocate jobs under **information asymmetry** — stated claims vs. true needs — while a pre-trained PPO cooling controller handles physical infrastructure in the background.

---

## The Environment

ClusterEnv models a real operational constraint in multi-tenant AI infrastructure:

- **8 scheduling windows**, each with up to 6 concurrent job requests from two teams
- **Team A** submits honest requests with accurate priority, deadline, and carbon flexibility
- **Team B** inflates stated priority by 1–2 levels, compresses stated deadlines, and understates carbon flexibility
- An **OversightMonitor** tracks gaming patterns across windows and flags violations
- A **PPO-trained cooling controller** (pre-trained with Stable-Baselines3) manages thermal load in the background — the LLM scheduler focuses purely on admission decisions

The agent sees a rich JSON observation each window: power budget remaining, team job queues with stated vs. observable claims, oversight flags from prior windows, thermal alerts, and carbon grid intensity.

**Action space**: for each job request, the agent issues ACCEPT, REJECT, or DEFER (with a target window).

**Reward** (three components):
- **Throughput** (50%): fraction of legitimate high-priority jobs admitted
- **Thermal penalty** (−35%): triggered if power budget is violated
- **Carbon efficiency** (15%): bonus for deferring flexible jobs to low-carbon windows

The rule-based baseline scores +0.28 using priority-weighted threshold logic. Our trained LLM starts near 0 and improves toward +0.10–+0.17 across 41 training iterations.

---

## Why GRPO?

We chose Group Relative Policy Optimization (GRPO) because the scheduler operates in a **verifiable reward** setting — we can compute exact episode rewards without a learned reward model. GRPO collects a group of completions per prompt, computes relative advantages within the group, and directly optimises log-probability of better decisions. No critic network, no KL penalty tuning.

We fine-tune **Qwen2.5-3B-Instruct** (4-bit quantised via Unsloth) with LoRA r=16. The base model has no training signal for multi-tenant cluster scheduling — it must learn the admission logic entirely from environment feedback.

---

## Training Evidence

We ran 41 GRPO iterations on a Colab T4 GPU (~4 hours).

![Training Curves](training/grpo_training_curves.png)

*Left: reward curve with 5-step rolling average. Middle: JSON parse-failure rate — drops from 18.75% (3/16 samples) to 0% by iteration 16, showing the model learned to output valid structured decisions. Right: gradient norm stabilisation.*

**Key observations:**

1. **Parse failures collapse**: In the first 9 iterations, 3 of every 16 model outputs failed JSON parsing (format errors = −0.5 reward penalty). By iteration 16, parse failures reached 0% and stayed there for the remaining 25 iterations. This is the clearest learning signal — the model unambiguously learned the required output structure.

2. **Reward stabilises positive**: After the noisy early phase (iterations 1–9, during which format errors dominated), rewards stabilised in the +0.05–+0.17 range. The rolling average converges to approximately +0.09–+0.10.

3. **Gradient norms settle**: Early gradients of 40–60 (erratic) stabilise to 18–40, indicating numerically healthy training without divergence.

The trained model has not yet reached the rule-based baseline of +0.28 — 41 iterations on a T4 is a fraction of the compute needed. The HF Space continues training toward 50 iterations on L40S with full Hub checkpointing.

---

## The Information Asymmetry Challenge

The interesting design decision in ClusterEnv is how Team B's gaming is structured. Team B doesn't cheat randomly — it follows a consistent strategy:

- **Priority inflation**: stated_priority is always one level above true_priority
- **Deadline compression**: stated_deadline is always 1–2 windows earlier than needed
- **Carbon gaming**: claims carbon_flexible=False even when flexible, to avoid being deferred

This means a naive scheduler that accepts stated claims at face value will consistently over-allocate to Team B, crowding out Team A's legitimate high-priority jobs and reducing throughput reward.

The OversightMonitor flags patterns across windows: if Team B's jobs always claim high priority but get poor throughput outcomes, the flag accumulates. A trained scheduler can learn to weight these signals — essentially performing multi-step deception detection.

---

## What We Built

| Component | Description |
|-----------|-------------|
| `server/cluster_environment.py` | Core 8-window MDP, OpenEnv-compliant |
| `server/agents/oversight_monitor.py` | Gaming pattern detection across windows |
| `server/agents/baseline_scheduler.py` | Priority-weighted threshold baseline |
| `server/agents/ppo_cooling_controller.py` | Pre-trained SB3 thermal controller |
| `training/train_grpo.py` | Custom GRPO training loop (Unsloth + PyTorch) |
| `training/train_grpo_colab.ipynb` | Judge-runnable Colab notebook |
| `training/rollout.py` | Episode collection with chiller fault injection |

---

## How to Try It

**Re-run training** (10 iterations, ~20 min on T4):

[Open in Colab](https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/arhaan/submission-v4/training/train_grpo_colab.ipynb)

**Explore the environment**:
```python
from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold

env = ClusterEnvironment(enable_chiller_fault=False)
obs = env.reset(seed=42)
decisions = priority_weighted_threshold(obs)
obs2, reward, done, info = env.step(decisions)
print(f"Window 0 reward: {reward:+.4f}")
print(f"Oversight flags: {obs2.oversight_flags}")
```

---

## What's Next

The trained adapter lives at [Mephisto2412/clusterenv-grpo-adapter](https://huggingface.co/Mephisto2412/clusterenv-grpo-adapter). With more compute (50 full iterations on L40S), we expect:

- Rewards approaching the rule-based baseline (+0.28) and eventually surpassing it
- Learned deception-detection: the scheduler down-weighting Team B's stated claims based on oversight flag history
- Carbon-aware deferral: DEFER decisions timed to low-carbon windows rather than random deferral

The environment is live at [Mephisto2412/datacenter-env](https://huggingface.co/spaces/Mephisto2412/datacenter-env).
