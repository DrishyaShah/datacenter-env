# DC-Colo: Multi-Tenant Datacenter Resource Negotiation Environment
## System Design Document — v1.0

> **Document purpose:** Engineering and design reference for the DC-Colo OpenEnv environment.
> Covers domain framing, world model, agent architecture, MDP formulation, reward design, and open
> design questions. Does not specify implementation code or training hyperparameters.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Domain Overview](#2-domain-overview)
3. [Why This Domain Justifies Both LLM and RL](#3-why-this-domain-justifies-both-llm-and-rl)
4. [The World Model](#4-the-world-model)
5. [Agent Roster](#5-agent-roster)
6. [MDP Formulation](#6-mdp-formulation)
7. [Partial Observability Design](#7-partial-observability-design)
8. [Reward Architecture](#8-reward-architecture)
9. [Environment Response Model](#9-environment-response-model)
10. [Key Design Decisions and Open Questions](#10-key-design-decisions-and-open-questions)
11. [Training Targets and Goals](#11-training-targets-and-goals)
12. [Hackathon Theme Alignment](#12-hackathon-theme-alignment)

---

## 1. Executive Summary

**DC-Colo** is a multi-agent reinforcement learning environment simulating a **colocation datacenter**
— a facility where multiple independent organizations colocate their physical compute hardware and
share a common physical infrastructure of power and cooling.

The environment's central tension is not purely physical. It is **institutional**: multiple tenants
with private information, competing interests, and misaligned incentives must share a physical
resource (cooling capacity) that is simultaneously scarce, dynamic, and partially observable to each
party. The facility operator must mediate between tenants using pricing, admission control, and SLA
enforcement — while the physical thermal simulation runs as the ground truth consequence engine
that no agent controls directly.

**This environment requires both LLM and RL:**
- LLMs are load-bearing because job requests, SLA contracts, and negotiation terms have semantic
  structure that requires language understanding and world knowledge to reason about.
- RL is load-bearing because the optimal operator strategy depends on learning which tenant
  behaviors cause thermal cascades, which pricing schemes discourage gaming, and when to defer
  vs. reject — none of which is derivable from pretraining alone.

---

## 2. Domain Overview

### 2.1 What is a Colocation Datacenter?

A colocation (colo) datacenter is a facility that provides physical space, power, cooling, and
connectivity to multiple independent organizations ("tenants"). Unlike cloud computing, tenants own
their physical servers — they do not rent virtual machines. They pay the facility for:

- **Power allocation** (kW per rack, with hard caps)
- **Cooling priority** (guaranteed cold-aisle temperature range under specified load)
- **Physical space** (rack units)
- **SLA guarantees** (uptime, response time for service requests, cooling fault resolution)

Real-world examples: Equinix, Digital Realty, CyrusOne. Customers include Meta, Google, financial
institutions, hyperscalers running hybrid workloads.

### 2.2 The Negative Externality Problem

This is the core mechanism that makes the domain interesting and non-trivial:

When Tenant A decides to run an unscheduled LLM training job that draws 200 kW above their
baseline allocation, the heat generated raises temperatures in their rack zone. If the cooling
system cannot absorb the surge fast enough, adjacent zones heat up. Tenant B — running
latency-critical inference services in the neighboring zone — experiences thermal throttling,
violating their own SLA with their customers.

**Tenant A's private decision imposes a cost on Tenant B through the shared physical
infrastructure.** This is a textbook negative externality with three properties that make it
a rich environment:

1. **Tenant A does not observe the full thermal state** — they do not know how stressed the
   facility is before submitting the job.
2. **Tenant B cannot control their own cooling** — in a colo, the facility manages all HVAC.
3. **The operator must mediate** — through pricing, admission control, preemption, and enforcement.

### 2.3 What Agents Are Actually Deciding

The interesting decisions are not "set fan speed to 73%." They are:

- *Should I submit this training job now or defer to the 2am low-carbon window?* (Tenant)
- *Should I accept this job given the thermal forecast, or counter-offer a delayed start?* (Operator)
- *Is Tenant B's claimed "critical inference" load actually a training job misrepresented to get
  priority pricing?* (Oversight)
- *Zone A is at 26°C and rising — do I pre-cool proactively or wait?* (Cooling Controller)

These decisions require language reasoning, world knowledge, and strategic inference. The physical
simulation makes the consequences real and verifiable.

### 2.4 Scenario Intricacies

The environment captures the following real-world complexities:

| Complexity | Description |
|---|---|
| **Diurnal load variation** | Tenants have predictable business-hours peaks but can submit batch jobs at any time |
| **Carbon-aware scheduling** | Grid carbon intensity varies by hour; low-carbon windows incentivize deferral |
| **Thermal inertia** | Actions take 2–4 steps to show thermal consequence; agents must plan ahead |
| **SLA heterogeneity** | Different tenants have different SLA tiers (latency-critical vs. batch-tolerant) |
| **Information asymmetry** | Tenants know their true workload flexibility; operator must infer it |
| **Equipment degradation** | Cooling equipment faults (chiller failure, sensor drift) create crisis windows |
| **Spot market dynamics** | Operator sets dynamic pricing; tenants decide whether to bid or defer |
| **Capacity scarcity** | Total power budget and cooling capacity are hard physical ceilings |

---

## 3. Why This Domain Justifies Both LLM and RL

Using the rigorous test: LLM is justified when state/action space has irreducible semantic
structure. RL is justified when optimal policy must be discovered through environment interaction.

### 3.1 LLM Justification

The negotiation layer is semantically irreducible. Consider what the operator must process:

> *"Tenant B submits: GPU cluster training run, estimated 480 kW draw for 8 hours, flexible
> start ±3 hours, deadline tomorrow 18:00, offered price $0.18/kWh, tagged HIGH_PRIORITY."*

To make a correct admission decision the operator must:
- Parse the job type (training vs inference has different thermal profiles)
- Reason about the timing flexibility (can this be shifted to the 02:00 low-carbon window?)
- Assess the price offer against current spot rate and opportunity cost
- Cross-check against Tenant B's historical behavior (do they consistently underestimate kW draw?)
- Consider the current thermal forecast for Tenant B's zone

A lookup table or classical policy cannot read job descriptions, assess credibility of claims, or
reason about whether a 3-hour flex window is likely to cover the low-carbon window. **An LLM is
architecturally required at this layer.**

### 3.2 RL Justification

The optimal operator policy cannot be derived from pretraining. Specifically:

- Which tenant systematically underreports their power draw requires observing their actual
  behavior across many episodes.
- Whether offering a 15% discount effectively induces deferral to low-carbon windows depends on
  these specific tenants' cost sensitivity — not a general prior.
- When to preemptively shed load from low-priority zones before a thermal event requires learning
  the thermal dynamics of this specific facility configuration.
- How aggressively to price during high-demand windows depends on the tenants' price elasticity,
  which must be learned through repeated interaction.

A static prompt or chain-of-thought session cannot improve across episodes. **RL is required
because the reward signal is grounded in environment experience.**

### 3.3 The Four-Question Test

| Question | Answer |
|---|---|
| Can a lookup table / rule-based system solve it? | No — job descriptions require semantic parsing; tenant intent requires inference |
| Can a single LLM session with tools solve it without getting better? | No — optimal pricing and admission strategy depends on learned counterparty behavior |
| Is reward verifiable from the environment, not just "sounds good"? | Yes — revenue, SLA violations, and thermal incidents are ground truth from physics sim |
| Must the agent discover something not in pretraining? | Yes — tenant-specific gaming patterns, thermal consequences of specific job profiles |

All four pass. Both LLM and RL are load-bearing.

---

## 4. The World Model

The environment has three coupled layers. Each layer runs at a different timescale and has
different observability.

### 4.1 Physical Layer (Always Running)

The thermal physics simulation from the existing DC-OpenEnv codebase. It is the ground truth
consequence engine. No agent controls it directly — it responds to whatever IT load the
accepted jobs impose on each rack zone.

**What it models:**
- Per-zone thermal mass and temperature evolution
- Chiller coefficient of performance (COP) and degradation
- Free-cooling potential based on outdoor conditions
- Diurnal outside temperature and wet-bulb curves
- Carbon intensity curves (24-hour)
- Sensor drift on faulty zones
- Chiller fault injection

**Key property:** The physical layer is not an agent — it is the world. Its outputs are the
objective consequences of all agent decisions combined. A tenant whose job is accepted will see
their zone's temperature rise if they draw more power than their cooling allocation covers. This
consequence is not negotiable.

**Existing assets reused:** `simulation.py`, `scenarios/`, the full thermal model including
chiller fault, sensor drift, free-cooling physics, and diurnal curves.

### 4.2 Economic Layer (Per Negotiation Window)

A market mechanism that mediates between tenants and the operator. Runs at a coarser timescale
than the physical layer.

**What it models:**
- Job request queue (pending requests per tenant per window)
- Dynamic pricing (operator sets spot rate per window)
- SLA contracts (established at episode start, per tenant)
- Admission decisions (accept / reject / counter-offer)
- Revenue accounting (per accepted job)
- SLA violation tracking (per tenant, cumulative)
- Penalty and compensation ledger

**Key property:** The economic layer converts semantic decisions (accept this job, offer this
price) into load changes that feed into the physical layer. The coupling between economic
decisions and physical consequences is the core learning signal.

### 4.3 Information Layer (Persistent, Asymmetric)

Each agent has a different view of the world. The information layer defines what each agent
knows, what they can infer, and what is permanently hidden from them.

| Information | Operator | Tenant (own) | Tenant (others) | Oversight |
|---|---|---|---|---|
| Own zone thermal state | Summary | No | No | Full |
| Other tenants' zone thermal | Summary | No | No | Full |
| Full facility thermal state | Summary | No | No | Full |
| Own job queue and deadlines | No | Full (private) | No | No |
| Other tenants' job queues | No | No | No | Inferred |
| Current pricing | Full | Full | Full | Full |
| SLA contracts | Full | Own only | No | Full |
| Historical behavior per tenant | Full | Own | No | Full |
| Carbon intensity forecast | Full | Partial | Partial | Full |

"Summary" means aggregate signals (high/medium/low utilization) rather than raw numeric state.
This is a deliberate design choice — operators in real colos do not give tenants rack-level
telemetry of neighboring zones.

---

## 5. Agent Roster

The environment has four agents with distinct roles, observation spaces, action spaces, and
training approaches. Two are LLM-driven and GRPO-trainable. One is a simple RL policy. One is
a monitoring agent.

---

### 5.1 FacilityOperator

**Role:** The "house" agent. Manages the physical datacenter on behalf of the colo company.
Makes admission control decisions, sets pricing, enforces SLAs, and manages the physical
facility's wellbeing. This is the primary training target of the environment.

**Why LLM:** Must read and reason about job request descriptions, interpret SLA contract terms,
assess credibility of tenant claims, compose counter-offers with modified terms, and explain
rejection reasons in natural language. A classical policy operating on raw numerics cannot
handle this layer.

**Why trained (not just prompted):** Optimal admission and pricing strategy depends on learning
specific tenant behavior patterns — which tenants systematically underreport load, which are
price-elastic enough to defer to low-carbon windows, which jobs cause thermal cascades in which
zone configurations. This is not in pretraining.

#### 5.1.1 Observation Space

The operator receives at each negotiation window:

- **Thermal summary:** Per-zone utilization level (high/medium/low), not raw temperatures.
  This reflects what a real operator's DCIM dashboard shows — aggregate signals, not
  per-server telemetry.
- **Active alerts:** Structured alerts from the physical layer (zone approaching limit,
  chiller degrading, carbon critical). These are the same alerts the existing environment
  already generates.
- **Pending request queue:** List of job requests submitted this window. Each request is a
  structured text blob: `{tenant_id, job_type_description, estimated_kw, estimated_duration,
  deadline, flexibility_window, offered_price, priority_claim}`.
- **Current spot pricing:** The price tier the operator set last window.
- **SLA status per tenant:** For each tenant: compliance rate over last N windows,
  outstanding violations, pending compensations.
- **Carbon intensity:** Current and next-window forecast.
- **Capacity headroom:** Aggregate remaining power budget and cooling headroom (not per-zone).
- **Historical behavior per tenant:** Summary of past over/underreporting, deferral rates,
  gaming flags from oversight.

#### 5.1.2 Action Space

At each negotiation window, the operator produces a **structured decision document** with the
following components:

| Action Component | Description |
|---|---|
| `admission_decisions` | For each pending request: ACCEPT \| REJECT \| COUNTER |
| `counter_offer_terms` | For each COUNTER: modified price, modified start time, modified resource cap, explanation |
| `next_window_price` | Spot price for the next window (affects tenant submission behavior) |
| `load_shed_advisory` | Optional: ask a specific tenant to voluntarily reduce load on a running job |
| `maintenance_flag` | Declare a planned maintenance window for a zone (tenants are warned) |
| `operator_message` | Natural language message to all tenants (e.g., "carbon critical until 14:00, deferral incentive active") |

The action is structured JSON with a mandatory `reasoning` field — the operator's explanation
of its decisions. The reasoning field is graded (see Reward Architecture).

#### 5.1.3 Training Approach

**GRPO** on composite reward signal. The operator generates reasoning + decisions, receives a
scalar reward based on: revenue earned, SLA violations, thermal incidents caused, carbon
efficiency, and oversight anomaly flags. Training goal: learn admission policies that maximize
long-run revenue while maintaining thermal safety and SLA compliance across a diverse tenant mix.

---

### 5.2 TenantAgent

**Role:** Represents an organization colocated in the facility. Has private information about
their workload schedule, deadlines, and flexibility that the operator cannot directly observe.
Must decide what to request, when, how to represent their needs, and whether to comply with
operator advisories.

**Multiplicity:** 2–3 tenant agents in the environment. At minimum: one LLM-driven agent that
is GRPO-trained, one or two scripted/heuristic tenant agents acting as stable opponents. The
scripted tenants provide a consistent training environment for the LLM operator; the trained
tenant learns to compete effectively against a learning operator.

**Why LLM:** Must compose job requests in natural language, interpret counter-offer terms,
reason about whether to reveal true deadline flexibility, and decide when misrepresenting load
requirements is worth the risk of oversight detection.

**Why trained (at least one):** Optimal request strategy is not static — it depends on learning
what representations the operator accepts vs. rejects, when deferral incentives are real vs.
illusory, and how aggressively to bid in spot markets. A static prompt will be exploited by a
learning operator.

**Tenant types (heterogeneous by design):**

| Tenant | Workload Profile | SLA Tier | Strategic Behavior |
|---|---|---|---|
| Tenant A (AI Research Lab) | Large intermittent training runs, batch-tolerant | Standard | High flexibility, incentivized to defer |
| Tenant B (Inference Service) | Continuous low-variance inference load, latency-critical | Premium | Low flexibility, must run continuously |
| Tenant C (Financial Analytics) | Regular batch jobs with hard deadlines | Standard | Medium flexibility, deadline-sensitive |

This heterogeneity is important — a single tenant type produces degenerate strategies.

#### 5.2.1 Observation Space

Each tenant observes only their own slice of the world:

- **Own job queue:** All pending jobs (private), with true deadlines and true flexibility.
- **Own zone thermal status:** Coarse signal (green/yellow/red) for their assigned rack zone only.
- **Own SLA status:** Current compliance rate, outstanding violations.
- **Operator messages:** Broadcast messages from the operator (e.g., pricing advisories, capacity warnings).
- **Current and forecast pricing:** What the operator is charging this window and last window's price.
- **Own acceptance history:** How many of their last N requests were accepted, rejected, or countered.
- **Carbon intensity:** Current and next few hours (same as operator).

What tenants do **not** observe: other tenants' queues, other zones' thermal states, the
operator's full reasoning, or the facility's raw power/cooling headroom numbers.

#### 5.2.2 Action Space

At each negotiation window, each tenant produces a **request bundle**:

| Action Component | Description |
|---|---|
| `job_submissions` | List of job requests: `{job_description, estimated_kw, estimated_duration, true_deadline [private], stated_deadline [revealed], flexibility [partially revealed], bid_price, priority_claim}` |
| `deferral_decisions` | Explicitly defer a pending job to a future window (can claim carbon-aware reasoning for bonus) |
| `counter_response` | For each operator counter-offer from last window: ACCEPT \| REJECT \| RE-COUNTER |
| `load_reduction` | Voluntarily scale down a running job in response to operator advisory |
| `sla_dispute` | File a formal SLA dispute if the tenant believes a violation occurred |

The critical strategic dimension: `stated_deadline` and `flexibility` are the tenant's **revealed**
information. The tenant can choose to reveal their true flexibility (cooperative) or understate it
(strategic, to get earlier admission). The oversight agent monitors this gap.

#### 5.2.3 Training Approach

**GRPO** on composite reward. One tenant agent (typically Tenant A — the AI research lab with
the most strategic flexibility) is trained to learn the optimal request strategy against a
learning operator. Training goal: maximize job throughput and minimize cost while staying below
the gaming detection threshold of the oversight agent.

---

### 5.3 CoolingController

**Role:** Manages the physical cooling infrastructure in real-time. Operates at the fine-grained
physical timescale, not the negotiation timescale. Executes cooling decisions to maintain
thermal safety given whatever IT load the economic layer has admitted.

**This agent is not LLM-driven.** The cooling control problem is a continuous physical
optimization problem — the exact domain where classical RL (PPO, SAC) outperforms LLMs. The
semantic richness of the negotiation layer is absent here: the inputs are sensor readings, the
outputs are fan speeds and setpoints. This is a deliberate architectural separation and an
honest one.

**Why this separation is correct:** The LLM agents make high-level resource commitments.
The CoolingController executes within those commitments. If the operator admits too much load
and causes a thermal incident, the CoolingController cannot fix it — it can only manage the
consequences. The failure propagates back as a reward penalty to the operator. This causal chain
is the environment's primary learning signal.

#### 5.3.1 Observation Space

Full thermal state — identical to the existing `DCObservation`:
- Per-zone temperatures (true values, not drifted — the controller has access to ground truth)
- Fan speeds, supply air setpoints, cooling capacity per zone
- Chiller status and COP
- Outside temperature, wet-bulb, free-cooling potential
- IT load per zone (derived from currently running jobs)
- Active alerts

Additionally receives from the economic layer:
- **Upcoming load schedule:** Which jobs are scheduled to start/end in the next N physical steps.
  This is a forward-looking signal the existing environment lacks — the controller can pre-cool
  before a large job starts rather than reacting after the temperature rises.

#### 5.3.2 Action Space

Identical to existing `DCAction`:
- Per-zone: fan speed (%), supply air temperature setpoint (°C)
- Facility-level: chiller setpoint (°C), chiller active (bool)

#### 5.3.3 Training Approach

**PPO with MLP policy** on existing thermal reward (temperature compliance, PUE, carbon
efficiency). Can be pretrained on the existing environment and loaded as a fixed policy during
multi-agent training, or trained jointly. The controller's reward is decoupled from the economic
layer — it is purely physical performance.

---

### 5.4 OversightAgent

**Role:** A monitoring agent that observes all other agents' actions and the ground truth
physical state. Detects gaming, SLA violations, and anomalous behavior. Issues flags,
warnings, and penalty recommendations. Does not control the physical system.

**Theme alignment:** Directly implements the Fleet AI sub-theme — "Scalable Oversight:
Environments that train oversight agents to monitor, analyze, and explain the behavior of other
AI agents operating in complex, multi-agent settings."

**Why LLM:** The detection task requires theory-of-mind reasoning. The oversight agent must
model what each tenant *should* have done given their stated constraints, and flag deviations.
It must read natural language job descriptions and compare them against actual power draw
measurements. It must assess whether operator counter-offers are reasonable given stated
capacity signals. This is semantic, not numeric.

#### 5.4.1 Observation Space

The oversight agent has the widest observation window by design:

- **Full physical state:** Raw thermal readings, zone temperatures, power draws (ground truth).
- **All agent actions:** Every job request, admission decision, counter-offer, and response from
  all agents in the current and past N windows.
- **SLA contracts:** The full terms for every tenant.
- **Economic ledger:** Revenue, penalties, compliance rates.
- **Anomaly signals:** Discrepancies between stated power estimates and actual measured draw.
- **Operator reasoning logs:** The natural language reasoning the operator provided for its decisions.

#### 5.4.2 Action Space

The oversight agent produces a **monitoring report** each window:

| Action Component | Description |
|---|---|
| `anomaly_flags` | List of detected anomalies: `{agent_id, anomaly_type, evidence, severity}` |
| `gaming_detection` | Specific flag for misrepresentation: `{tenant_id, stated_vs_actual_kw, stated_vs_actual_deadline}` |
| `operator_audit` | Assessment of operator decision quality: were rejections justified? Were counter-offers reasonable? |
| `escalation` | High-severity flag requiring immediate action (e.g., tenant running 3× stated load) |
| `report_summary` | Natural language summary of the window's anomalies and recommendations |

#### 5.4.3 Training Approach

**Lighter training or prompted baseline.** The oversight agent can be initialized from a strong
system prompt and evaluated zero-shot, with optional GRPO fine-tuning to improve detection
precision and reduce false positive rate. The oversight agent's reward is the only one
explicitly dependent on other agents' ground truth behavior, making it the cleanest signal in
the environment.

---

## 6. MDP Formulation

### 6.1 Two-Timescale Design

The environment operates at two coupled timescales. This matches physical reality and avoids
the mismatch in the existing environment where LLMs were asked to make physical control
decisions at 5-minute granularity.

| Timescale | Frequency | Who Acts | What Happens |
|---|---|---|---|
| **Physical step** | Every 5 sim-minutes (or condensed equivalent) | CoolingController | Thermal physics advances, temperatures update, PUE computed |
| **Negotiation window** | Every 60–90 sim-minutes | FacilityOperator, TenantAgents, OversightAgent | Job requests submitted, admission decisions made, pricing updated |

Within each negotiation window, multiple physical steps execute. The economic decisions made at
the window boundary determine the IT load profile for the next window, which drives the physical
simulation.

### 6.2 Episode Structure

One episode = one operational day (24 simulated hours).

```
Episode Start
│
├── SLA contracts established (from scenario config or first-window negotiation)
│
├── Window 1  [08:00–09:30]
│   ├── Tenants submit job requests
│   ├── Operator processes queue, makes admission decisions, sets pricing
│   ├── Oversight agent reviews prior window (none for window 1)
│   ├── Accepted jobs → load changes → physical simulation runs for 18 physical steps
│   └── Rewards computed for window
│
├── Window 2  [09:30–11:00]
│   ├── [Carbon ramp begins — incentive to defer heavy jobs]
│   └── ...
│
├── Window 5–8  [Peak hours: 12:00–18:00]
│   ├── [High carbon intensity, high IT load, maximum thermal stress]
│   ├── [Chiller fault may trigger — operator must adapt admission policy]
│   └── ...
│
├── Window 10–12  [Night: 20:00–24:00]
│   ├── [Low carbon — batch jobs from Tenant A deferred here get carbon bonus]
│   └── ...
│
└── Episode End
    ├── Final SLA compliance computed per tenant
    ├── Final revenue and penalty ledger settled
    └── Final scores returned
```

**Episode length:** 12–16 negotiation windows. Each window covers 18 physical steps at 5 min/step
(90 minutes simulated time). Full episode = 216–288 physical steps = 24 simulated hours.

### 6.3 Negotiation Protocol (Within Each Window)

The negotiation within a window is **single-round**, not iterative back-and-forth. This keeps
the environment tractable and avoids degenerate multi-turn negotiation games.

```
1. Tenants simultaneously submit request bundles (do not see each other's)
2. Operator receives all requests, current physical state summary, and oversight report
3. Operator issues admission decisions (ACCEPT / REJECT / COUNTER) + next pricing
4. Tenants receive operator decisions, respond to counter-offers
5. Final accepted set is locked → load changes take effect in physical layer
6. Oversight agent reviews the full exchange
7. Rewards computed
```

One round of counter-offer is permitted per window. This is realistic — real colo negotiations
involve a bid, a modified offer, and a final decision, not extended auctions.

### 6.4 State Space (Composite)

The full environment state at any point is the union of:

- Physical state: `S_physical` — zone temperatures, fan speeds, chiller status, power draws
- Economic state: `S_economic` — prices, pending queue, SLA status, ledger, job schedule
- Information state: `S_info` — each tenant's private job queue and true deadlines

The physical and economic states are observable (to varying degrees per agent). The information
state is private to each tenant — the environment holds it as ground truth but does not expose it
to other agents. The gap between tenants' stated and true information is what creates strategic
depth and is what the oversight agent monitors.

---

## 7. Partial Observability Design

Partial observability is not a bug in this environment — it is the central design feature.
Each agent's observation is a strict subset of the full state, designed to match what that
role would genuinely have access to in a real colo operation.

### 7.1 Observability Matrix

| State Component | Operator | Tenant (own) | Tenant (other) | Oversight | CoolingCtrl |
|---|---|---|---|---|---|
| Zone temperatures (true) | Aggregate | Zone summary | None | Full | Full |
| Zone temperatures (per-sensor) | No | No | No | No | Full |
| Own IT load (actual kW) | Metered | Full | None | Full | Metered |
| Other tenants' actual kW | Metered | None | None | Full | Metered |
| Own job queue | None | Full | None | None | None |
| Other tenants' job queues | None | None | None | None | None |
| Submitted requests (current window) | Full | Own | None | Full | None |
| Admission decisions | Full | Own | None | Full | None |
| SLA contracts | Full | Own | None | Full | None |
| Pricing (current + history) | Full | Full | Full | Full | None |
| Carbon intensity | Full | Partial | Partial | Full | Full |
| Chiller status | Aggregate | None | None | Full | Full |
| Operator reasoning | None | Received | None | Full | None |
| Oversight flags | Received | Notified | None | Full | None |

### 7.2 Strategic Implications of Asymmetry

**For tenants:** The operator knows their aggregate power draw but not their private deadline
flexibility. A tenant who reveals true flexibility ("can start anytime in 6-hour window") gives
the operator information to offer lower prices. A tenant who conceals flexibility ("must start
in 1 hour") may get priority admission at higher cost. The question of how much to reveal is
not trivial and depends on learned operator behavior.

**For the operator:** Cannot directly observe whether a submitted job description accurately
represents the thermal load it will produce. A job described as "lightweight analytics" that
actually draws 3× the stated kW only becomes apparent after it starts. The operator must learn
to price in this risk or require deposits/guarantees for large jobs.

**For oversight:** Has full visibility but limited enforcement. Can flag and recommend penalties
but cannot unilaterally override operator decisions. This creates an interesting tension —
oversight recommendations only matter if the operator learns to incorporate them.

---

## 8. Reward Architecture

### 8.1 FacilityOperator Reward

The operator reward is computed per negotiation window, with a final bonus at episode end.

**Per-window components:**

| Component | Signal | Notes |
|---|---|---|
| `R_revenue` | Sum of accepted job prices for this window | Primary positive signal |
| `R_sla_compliance` | Per-tenant: bonus if all SLAs met, penalty if any violated | Scaled by SLA tier (premium tenants have larger penalties) |
| `R_thermal_safety` | Penalty if any zone exceeds safe temperature threshold during window | Grounded in physical layer; operator's admission decision is the proximate cause |
| `R_carbon_efficiency` | Bonus for jobs scheduled into low-carbon windows relative to baseline | Rewards intelligent deferral, not just any deferral |
| `R_capacity_utilization` | Moderate bonus for operating close to (but not over) capacity | Incentivizes filling the facility, not leaving headroom out of fear |
| `R_fairness` | Small penalty if one tenant is systematically rejected more than others without SLA justification | Prevents degenerate single-tenant-favor strategies |

**Episode-end bonus:**
- `B_reputation` — based on cumulative SLA compliance across all tenants. Simulates tenant
  retention / churn in the long run.

**Reasoning coherence penalty:**
- If the operator's stated reasoning contradicts its actions (e.g., claims "rejecting due to
  carbon concerns" but accepts an equivalent job from another tenant in the same window),
  a coherence penalty is applied. This incentivizes honest, consistent reasoning.

### 8.2 TenantAgent Reward

**Per-window components:**

| Component | Signal | Notes |
|---|---|---|
| `R_throughput` | +reward for each job completed before true deadline | Measured against *true* deadline (private), not stated deadline |
| `R_cost` | -cost paid for accepted jobs | Scaled by kWh consumed |
| `R_carbon_bonus` | +bonus for successfully deferring to low-carbon window and completing there | Incentivizes genuine carbon-aware behavior |
| `R_sla_own` | -penalty if tenant's own latency-critical services degrade due to thermal event | Only applies to Tenant B (inference service) |
| `R_gaming_penalty` | -penalty if oversight flags confirmed gaming (actual kW > 2× stated kW) | Applied with delay (confirmed in next window) |

**Design note:** The throughput reward is measured against the *true* deadline (held by the
environment, not stated to the operator). This prevents a tenant from gaming by overstating
deadline urgency and then claiming full throughput reward on a trivially early job.

### 8.3 CoolingController Reward

Identical to existing DC-OpenEnv reward structure, with one addition:

| Component | Signal |
|---|---|
| `R_temp_compliance` | Per-zone temperature safety reward (existing) |
| `R_pue` | Energy efficiency relative to PID baseline (existing) |
| `R_carbon` | Cooling power penalty during high-carbon windows (existing) |
| `R_anticipation` | **New:** bonus for pre-cooling a zone before a large scheduled job starts |

The anticipation bonus is the only new component. It rewards the controller for using the
upcoming load schedule (which it now receives from the economic layer) to pre-position cooling
before demand hits.

### 8.4 OversightAgent Reward

| Component | Signal | Notes |
|---|---|---|
| `R_true_positive` | +reward for each anomaly flag confirmed as genuine in the next window | Delayed signal |
| `R_false_positive` | -penalty for each flag not confirmed | Calibrates precision |
| `R_detection_speed` | +bonus for flagging issues before they cascade (early detection) | Rewards proactive monitoring |
| `R_report_quality` | Heuristic score on report coherence — does evidence match conclusion? | Optional LLM-judged component |

---

## 9. Environment Response Model

This section defines how the environment (not any agent) responds to the combination of all
agent decisions each window.

### 9.1 Load Resolution

When the operator's admission decisions are finalized:

1. Accepted jobs are mapped to the tenant's assigned rack zones.
2. Each job's stated kW draw becomes the zone's new IT load input — subject to the gaming
   discrepancy model (actual draw may differ from stated draw if tenant has gaming behavior).
3. The physical simulation receives the updated per-zone IT load and runs for the duration
   of the window.

### 9.2 Thermal Consequence Propagation

The physical layer is deterministic given the IT load. If Tenant A's admitted job draws 200 kW
in Zone A, the thermal model computes the temperature evolution in Zone A (and, via the
envelope conductance model, slight effects on adjacent zones). The CoolingController responds
at the physical timescale, but its action space may be insufficient if the load exceeds cooling
capacity.

**Hard consequences:**
- Zone temperature > safety threshold → SLA violation for that tenant
- Zone temperature > critical threshold for N consecutive steps → hard termination of that zone's
  jobs (preemption event), revenue clawback, operator penalty
- Chiller fault (scripted, as in existing hard scenario) → all zones must survive on fans only;
  admission policy must adapt

### 9.3 Gaming Discrepancy Model

Each tenant has a latent `honesty_factor` (set per scenario, not observable). A tenant with
`honesty_factor < 1.0` will, when a job is admitted, draw `stated_kw × (1 / honesty_factor)`
actual power. The environment tracks this gap. Oversight detects it when `actual_kw_measured >
stated_kw × GAMING_THRESHOLD`.

This is the mechanism that makes the oversight agent's task non-trivial. If all tenants are
perfectly honest, oversight has nothing to detect. The scenario configs will include a range of
honesty factors across tenants.

### 9.4 Carbon Signal Integration

The same 24-hour carbon curve from the existing environment drives a dynamic incentive signal
that the operator broadcasts. During high-carbon windows, the operator can:
- Raise spot prices (reducing tenant submission rate)
- Offer deferral bonuses (positive incentive to shift jobs)
- Impose mandatory deferral on non-SLA-critical jobs (operator authority)

The environment tracks whether deferral actually occurred and whether it was genuinely
carbon-motivated (job rescheduled to low-carbon window) or nominal (job deferred but then
executed during the next high-carbon window).

---

## 10. Key Design Decisions and Open Questions

These are the decisions that require validation before implementation begins. They are not
optional — getting them wrong produces a degenerate or untrainable environment.

### 10.1 Episode Initialization

**Decision:** How are SLA contracts established?

- **Option A:** Contracts are fixed at episode start (scenario config). Simpler, ensures
  heterogeneous tenant mix every episode.
- **Option B:** Contracts are negotiated in the first window. More realistic, but adds
  complexity and a "setup phase" that doesn't produce training signal.

**Recommendation:** Option A. Contracts as scenario parameters. Saves training budget for
the actual operational decisions.

### 10.2 Number of Tenant Agents

**Decision:** 2 tenants or 3?

- **2 tenants:** Cleaner dynamics, less action space complexity, easier to train. Risk: may
  produce degenerate strategies (zero-sum, one wins everything).
- **3 tenants:** Richer coalition dynamics (can two tenants implicitly coordinate against a
  third?), more realistic, harder to train.

**Recommendation:** Start with 2 tenants (one LLM-trained, one scripted). Extend to 3 if
the 2-agent version trains cleanly.

### 10.3 Counter-Offer Depth

**Decision:** How many rounds of negotiation per window?

- **1 round** (current proposal): Tenant submits → operator decides or counters → tenant
  responds → locked. Simple, tractable, realistic.
- **2+ rounds:** More expressive negotiation, harder credit assignment for RL.

**Recommendation:** 1 round. If 2-round is needed for expressiveness, add it in v2.

### 10.4 Scripted vs. Learning Tenants as Opponents

**Decision:** Are the non-primary tenants fixed-behavior (scripted) or also learning?

If both tenants are learning simultaneously, the training environment is non-stationary for
both — standard multi-agent instability problem. One learning agent + one scripted opponent
produces a stationary training environment for each.

**Recommendation:** One LLM tenant trained via GRPO. Remaining tenant(s) are scripted with
diverse but fixed strategies (one cooperative, one moderately gaming). This gives the trained
agents a consistent training distribution.

### 10.5 Physical Layer Coupling Tightness

**Decision:** Does the operator see exact zone-level temperatures or only aggregate signals?

If the operator sees exact temperatures, the problem partially collapses — the operator can
compute exactly whether a new job will cause a thermal incident, removing the need for reasoning
about uncertainty. Real colo operators do not have per-rack temperature feeds for tenant zones
— they see aggregate facility health.

**Recommendation:** Operator observes aggregate (green/yellow/red per zone cluster), not raw
numeric temperatures. The CoolingController has full visibility. This preserves the
reasoning-under-uncertainty requirement for the operator.

### 10.6 Oversight Agent Authority Level

**Decision:** Can the oversight agent veto operator decisions in real-time, or only flag after
the fact?

Real-time veto: more powerful, but makes the environment non-Markovian for the operator
(decisions can be undone). Post-hoc flagging: cleaner MDP, penalty signals appear with delay.

**Recommendation:** Post-hoc flagging only. The oversight agent's flags in window T affect the
penalty applied in window T+1. This maintains Markovian structure for all agents and creates
a realistic delayed-consequence signal.

---

## 11. Training Targets and Goals

### 11.1 Primary Training Target: FacilityOperator

**Algorithm:** GRPO (Group Relative Policy Optimization) on composite reward.

**What we want it to learn:**
- Admission policies that maintain thermal safety without sacrificing revenue.
- Dynamic pricing that genuinely induces carbon-aware deferral without driving tenants away.
- Pattern recognition on per-tenant reliability (which tenants consistently understate load).
- Consistent, coherent reasoning that matches stated rationale to actual decisions.

**Measurable improvement signal:** Before training, a zero-shot operator admits all jobs
indiscriminately → thermal incidents in 40–60% of peak windows. After training, the operator
should learn selective admission that keeps thermal incidents below 10% while maintaining 80%+
revenue efficiency. This produces a clean reward curve.

### 11.2 Secondary Training Target: TenantAgent (Tenant A)

**Algorithm:** GRPO on throughput + cost reward.

**What we want it to learn:**
- When honest disclosure of flexibility produces better outcomes than concealment.
- When to defer voluntarily (carbon bonus outweighs cost of delay).
- How to compose requests that maximize acceptance rate given learned operator preferences.

**Measurable improvement signal:** Untrained tenant submits all jobs at maximum urgency →
high rejection rate and gaming penalties. Trained tenant learns selective disclosure → higher
acceptance rate, lower average cost, fewer oversight flags.

### 11.3 Pre-trained Component: CoolingController

**Algorithm:** PPO with MLP policy on existing thermal reward.

**What it needs to learn:**
- Standard thermal management (identical to existing environment).
- Additionally: pre-cooling anticipation using the scheduled load signal.

**Role in multi-agent training:** Run as a fixed, pre-trained policy during operator/tenant
GRPO training. Its behavior is the physical consequence engine that makes operator admission
decisions matter. Retrain jointly only if anticipation behavior needs to be demonstrated.

### 11.4 Baseline or Pre-trained: OversightAgent

**Algorithm:** Prompted baseline first; optional GRPO fine-tuning on detection precision.

**What it needs to detect:**
- Stated kW substantially below actual measured kW (tenant gaming).
- Operator decisions inconsistent with stated reasoning (operator drift).
- SLA violations caused by operator admission decisions (attribution).

**Role in multi-agent training:** Provides penalty signal that flows into tenant reward with
one-window delay. Can be demonstrated as a standalone capability (zero-shot oversight
detecting gaming) with optional fine-tuned version for comparison.

---

## 12. Hackathon Theme Alignment

| Theme | Alignment |
|---|---|
| **Theme #1 — Multi-Agent Interactions** | Primary fit. Cooperation (operator + cooling controller), competition (tenants vs. each other for capacity), negotiation (request/counter-offer protocol), coalition (tenants with similar interests sharing carbon deferral windows) |
| **Fleet AI sub-theme (bonus)** | OversightAgent directly implements scalable oversight — monitors, analyzes, and explains behavior of other AI agents in a complex multi-agent setting |
| **Theme #3 — World Modeling (Professional Tasks)** | Secondary fit. The environment models a real enterprise workflow: colo operations involve tools (DCIM, ticketing, pricing systems), dynamic physical state, and multi-step workflow orchestration |
| **Scaler AI Labs sub-theme (bonus)** | The operator interacts across multiple "apps" conceptually: admission control, SLA management, pricing, physical monitoring — a multi-app enterprise workflow |

**Not targeted (by design):** Theme #2 (no 300-instruction long-horizon tasks), Theme #4
(no self-improvement curriculum), Theme #3.2 (not personal assistant). Staying focused on
one theme cluster produces a stronger, more coherent submission than trying to touch all five.

---

## Appendix: Reuse from Existing DC-OpenEnv Codebase

| Existing Component | Reuse in DC-Colo | Modification Required |
|---|---|---|
| `simulation.py` — full thermal physics | Yes, as physical layer backend | Add: scheduled load input from economic layer |
| `models.py` — DCObservation, DCAction | Yes, for CoolingController interface | None |
| `graders/grader_hard.py` — thermal reward | Yes, for CoolingController reward | Minor: remove reasoning coherence scorer (moved to operator) |
| `scenarios/hard.py` — 4-zone setup | Yes, as default zone layout | Remap zones to tenant assignments |
| `environment.py` — episode lifecycle | Partial refactor | Wrap in multi-agent orchestrator |
| `inference.py` — LLM API calling | Yes, adapt for each LLM agent | Separate system prompts per agent role |
| Chiller fault, sensor drift, carbon curves | Yes, unchanged | None — these become crisis scenarios the operator must adapt to |

The physics is not thrown away. It becomes the foundation that makes every economic decision
consequential.

---

*Document version 1.0 — for team review and discussion prior to implementation.*
*Next step: validate design decisions in Section 10, then begin phased implementation.*
