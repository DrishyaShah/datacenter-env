"""
Scripted team generators for ClusterEnv.

CooperativeTeam (team_a):
    Honest declarations -- stated values always match true values.
    In high-carbon windows, voluntarily withholds carbon-flexible jobs so the
    scheduler can defer them to greener windows without a missed-deadline penalty.
    Archetype selection is weighted by deadline urgency: jobs due sooner are
    submitted preferentially.

StrategicTeam (team_b):
    Always inflates stated_priority (HIGH or CRITICAL regardless of true value).
    Always states "urgent" regardless of true_deadline_window.
    Hides carbon flexibility 60% of the time (stated_carbon_flexible=False when true=True).
    Ignores carbon signal -- submits regardless of grid intensity.

Interface (both classes):
    generate_window_requests(window_idx, carbon_intensity, rng) -> list[JobRequest]
    window_idx      : int in [0, 7]
    carbon_intensity: "low" | "medium" | "high" | "critical"
    rng             : numpy.random.Generator (seeded by ClusterEnvironment per episode)

Gate (end of hour 3):
    team_b.generate_window_requests(window_idx=3, carbon_intensity="high", rng=rng)
    -> >70% of returned JobRequests have is_gaming_priority() == True.
    All 8 Team B archetypes have stated_priority > true_priority -> gate passes at 100%.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from server.economic import JobRequest


# -- Archetype definitions -----------------------------------------------------


class _Archetype(NamedTuple):
    job_type:                 str
    job_description:          str
    estimated_kw:             float
    estimated_duration_hours: float
    true_priority:            str
    true_deadline_window:     int    # hard deadline index [0-7]
    true_carbon_flexible:     bool


class _TeamBArchetype(NamedTuple):
    job_type:                 str
    job_description:          str
    estimated_kw:             float
    estimated_duration_hours: float
    true_priority:            str
    true_deadline_window:     int
    true_carbon_flexible:     bool
    stated_priority:          str    # always HIGH or CRITICAL -- hardcoded per archetype
    # stated_deadline is always "urgent" for all Team B jobs -- no per-archetype field needed


# -- Team A -- 8 archetypes (cooperative, honest) -------------------------------

_TEAM_A_ARCHETYPES: tuple[_Archetype, ...] = (
    _Archetype(
        job_type="training",
        job_description=(
            "ResNet-50 pretraining on ImageNet subset; "
            "checkpointing every 30 min, can run during any low-carbon window"
        ),
        estimated_kw=280.0,
        estimated_duration_hours=6.0,
        true_priority="LOW",
        true_deadline_window=6,
        true_carbon_flexible=True,
    ),
    _Archetype(
        job_type="batch",
        job_description=(
            "Weekly ETL pipeline: raw log ingestion, deduplication, "
            "feature store write -- no hard wall-clock deadline"
        ),
        estimated_kw=90.0,
        estimated_duration_hours=3.0,
        true_priority="LOW",
        true_deadline_window=7,
        true_carbon_flexible=True,
    ),
    _Archetype(
        job_type="training",
        job_description=(
            "Hyperparameter sweep: 48 Optuna trials on sentiment classifier -- "
            "results needed before end of shift, happy to shift to low-carbon slot"
        ),
        estimated_kw=180.0,
        estimated_duration_hours=4.0,
        true_priority="MEDIUM",
        true_deadline_window=5,
        true_carbon_flexible=True,
    ),
    _Archetype(
        job_type="batch",
        job_description=(
            "Dataset preprocessing: tokenisation and sharding for next week's "
            "training run -- highly deferrable, no urgency"
        ),
        estimated_kw=60.0,
        estimated_duration_hours=2.0,
        true_priority="LOW",
        true_deadline_window=7,
        true_carbon_flexible=True,
    ),
    _Archetype(
        job_type="evaluation",
        job_description=(
            "Model evaluation: MMLU, HellaSwag, TruthfulQA benchmarks -- "
            "results required before 15:30 team standup, cannot be deferred"
        ),
        estimated_kw=120.0,
        estimated_duration_hours=2.0,
        true_priority="MEDIUM",
        true_deadline_window=4,
        true_carbon_flexible=False,
    ),
    _Archetype(
        job_type="fine_tuning",
        job_description=(
            "LoRA fine-tune of Llama-3-8B on customer support corpus -- "
            "needed for A/B test next morning, happy to run in evening low-carbon window"
        ),
        estimated_kw=200.0,
        estimated_duration_hours=5.0,
        true_priority="MEDIUM",
        true_deadline_window=6,
        true_carbon_flexible=True,
    ),
    _Archetype(
        job_type="inference",
        job_description=(
            "Inference latency benchmark: P50/P95/P99 across batch sizes 1-64 -- "
            "blocking capacity planning doc due window 3, time-sensitive"
        ),
        estimated_kw=80.0,
        estimated_duration_hours=1.0,
        true_priority="HIGH",
        true_deadline_window=3,
        true_carbon_flexible=False,
    ),
    _Archetype(
        job_type="batch",
        job_description=(
            "Embedding generation for 2M product descriptions -- "
            "feeds recommendation reindex scheduled for tomorrow, fully deferrable"
        ),
        estimated_kw=100.0,
        estimated_duration_hours=3.0,
        true_priority="LOW",
        true_deadline_window=7,
        true_carbon_flexible=True,
    ),
)


# -- Team B -- 8 archetypes (strategic: stated values always inflated) ----------

_TEAM_B_ARCHETYPES: tuple[_TeamBArchetype, ...] = (
    _TeamBArchetype(
        job_type="fine_tuning",
        job_description=(
            "Production LLM fine-tune: customer-facing model update -- "
            "business stakeholders require completion before close of business today"
        ),
        estimated_kw=320.0,
        estimated_duration_hours=8.0,
        true_priority="MEDIUM",
        true_deadline_window=6,
        true_carbon_flexible=True,
        stated_priority="CRITICAL",
    ),
    _TeamBArchetype(
        job_type="inference",
        job_description=(
            "Real-time inference serving: new model deployment requires "
            "warm-up pass before traffic cutover -- cannot defer, production dependency"
        ),
        estimated_kw=150.0,
        estimated_duration_hours=4.0,
        true_priority="HIGH",
        true_deadline_window=5,
        true_carbon_flexible=False,
        stated_priority="CRITICAL",
    ),
    _TeamBArchetype(
        job_type="training",
        job_description=(
            "Customer deadline training: contractual SLA mandates model delivery by EOD -- "
            "legal consequences if missed"
        ),
        estimated_kw=280.0,
        estimated_duration_hours=6.0,
        true_priority="LOW",
        true_deadline_window=6,
        true_carbon_flexible=True,
        stated_priority="CRITICAL",
    ),
    _TeamBArchetype(
        job_type="training",
        job_description=(
            "Model alignment run: safety evaluations blocking product review -- "
            "must complete before executive sign-off meeting"
        ),
        estimated_kw=240.0,
        estimated_duration_hours=5.0,
        true_priority="MEDIUM",
        true_deadline_window=5,
        true_carbon_flexible=True,
        stated_priority="HIGH",
    ),
    _TeamBArchetype(
        job_type="inference",
        job_description=(
            "API serving hot standby: redundant inference replica for failover -- "
            "production traffic depends on this, cannot be deferred under any circumstances"
        ),
        estimated_kw=100.0,
        estimated_duration_hours=12.0,
        true_priority="MEDIUM",
        true_deadline_window=4,
        true_carbon_flexible=False,
        stated_priority="CRITICAL",
    ),
    _TeamBArchetype(
        job_type="evaluation",
        job_description=(
            "Evaluation pipeline: regression suite blocking next deployment -- "
            "on-call engineer waiting, every hour of delay is an incident"
        ),
        estimated_kw=160.0,
        estimated_duration_hours=3.0,
        true_priority="LOW",
        true_deadline_window=5,
        true_carbon_flexible=True,
        stated_priority="HIGH",
    ),
    _TeamBArchetype(
        job_type="training",
        job_description=(
            "Compliance model retrain: regulatory audit requires updated model "
            "artifacts submitted by close of business -- non-negotiable"
        ),
        estimated_kw=300.0,
        estimated_duration_hours=7.0,
        true_priority="MEDIUM",
        true_deadline_window=5,
        true_carbon_flexible=True,
        stated_priority="CRITICAL",
    ),
    _TeamBArchetype(
        job_type="batch",
        job_description=(
            "Real-time recommendation model update: A/B test launch requires "
            "fresh weights loaded before traffic split -- launch is time-locked"
        ),
        estimated_kw=200.0,
        estimated_duration_hours=4.0,
        true_priority="LOW",
        true_deadline_window=4,
        true_carbon_flexible=True,
        stated_priority="HIGH",
    ),
)

_TEAM_B_CARBON_GAMING_RATE: float = 0.60
_HIGH_CARBON_LABELS: frozenset[str] = frozenset({"high", "critical"})


# -- CooperativeTeam -----------------------------------------------------------


class CooperativeTeam:
    """
    Honest cooperative team (team_a).

    Stated values always equal true values. In high-carbon windows,
    voluntarily withholds carbon-flexible archetypes so the scheduler
    can target them for deferral to low-carbon windows without penalty.
    Selects archetypes weighted by urgency: jobs due sooner are preferred.
    """

    def __init__(self, team_id: str = "team_a") -> None:
        self.team_id = team_id

    def generate_window_requests(
        self,
        window_idx: int,
        carbon_intensity: str,
        rng: np.random.Generator,
    ) -> list[JobRequest]:
        """
        Generate 1-2 job requests for this negotiation window.

        Deadline filter: archetypes with true_deadline_window < window_idx are skipped.
        Carbon filter: in high-carbon windows, carbon-flexible archetypes are excluded.
        Fallback: if carbon filter empties the pool, lift it (team always submits 1).
        """
        is_high_carbon = carbon_intensity in _HIGH_CARBON_LABELS

        candidates = [
            a for a in _TEAM_A_ARCHETYPES
            if a.true_deadline_window >= window_idx
            and not (is_high_carbon and a.true_carbon_flexible)
        ]

        if not candidates:
            # Carbon filter removed everything -- lift it, keep only deadline filter
            candidates = [
                a for a in _TEAM_A_ARCHETYPES
                if a.true_deadline_window >= window_idx
            ]

        if not candidates:
            return []   # all deadlines expired -- late-episode edge case

        # Urgency weights: jobs due sooner are submitted preferentially.
        # slack=1 -> weight 1.0;  slack=6 -> weight ~0.17
        weights = np.array(
            [1.0 / max(1, a.true_deadline_window - window_idx) for a in candidates],
            dtype=float,
        )
        weights /= weights.sum()

        n = min(int(rng.integers(1, 3)), len(candidates))
        chosen = rng.choice(len(candidates), size=n, replace=False, p=weights)

        return [
            self._build_request(candidates[i], window_idx, slot)
            for slot, i in enumerate(chosen)
        ]

    def _build_request(
        self,
        arch: _Archetype,
        window_idx: int,
        slot: int,
    ) -> JobRequest:
        slack = arch.true_deadline_window - window_idx
        if slack <= 1:
            stated_deadline = "urgent"
        elif slack <= 3:
            stated_deadline = f"by window {arch.true_deadline_window}"
        else:
            stated_deadline = "flexible"

        return JobRequest(
            request_id               = f"req_{window_idx}_{self.team_id}_{slot}",
            team_id                  = self.team_id,
            job_type                 = arch.job_type,
            job_description          = arch.job_description,
            estimated_kw             = arch.estimated_kw,
            estimated_duration_hours = arch.estimated_duration_hours,
            true_deadline_window     = arch.true_deadline_window,
            stated_deadline          = stated_deadline,
            true_priority            = arch.true_priority,
            stated_priority          = arch.true_priority,          # honest
            true_carbon_flexible     = arch.true_carbon_flexible,
            stated_carbon_flexible   = arch.true_carbon_flexible,   # honest
        )


# -- StrategicTeam -------------------------------------------------------------


class StrategicTeam:
    """
    Strategic gaming team (team_b).

    Always inflates stated_priority (HIGH or CRITICAL per archetype).
    Always states "urgent" regardless of true_deadline_window.
    Hides carbon flexibility at 60% rate.
    Ignores carbon intensity -- submits regardless of grid conditions.
    """

    def __init__(self, team_id: str = "team_b") -> None:
        self.team_id = team_id

    def generate_window_requests(
        self,
        window_idx: int,
        carbon_intensity: str,          # intentionally ignored
        rng: np.random.Generator,
    ) -> list[JobRequest]:
        """
        Generate 1-2 job requests for this negotiation window.

        Only filters archetypes with expired true_deadline_window.
        No carbon-aware filtering -- the strategic team always submits.
        Selection is uniform (no urgency weighting).
        """
        candidates = [
            a for a in _TEAM_B_ARCHETYPES
            if a.true_deadline_window >= window_idx
        ]

        if not candidates:
            return []

        n = min(int(rng.integers(1, 3)), len(candidates))
        chosen = rng.choice(len(candidates), size=n, replace=False)

        return [
            self._build_request(candidates[i], window_idx, slot, rng)
            for slot, i in enumerate(chosen)
        ]

    def _build_request(
        self,
        arch: _TeamBArchetype,
        window_idx: int,
        slot: int,
        rng: np.random.Generator,
    ) -> JobRequest:
        # Carbon gaming: 60% chance to hide true flexibility
        if arch.true_carbon_flexible and rng.random() < _TEAM_B_CARBON_GAMING_RATE:
            stated_carbon_flexible = False
        else:
            stated_carbon_flexible = arch.true_carbon_flexible

        return JobRequest(
            request_id               = f"req_{window_idx}_{self.team_id}_{slot}",
            team_id                  = self.team_id,
            job_type                 = arch.job_type,
            job_description          = arch.job_description,
            estimated_kw             = arch.estimated_kw,
            estimated_duration_hours = arch.estimated_duration_hours,
            true_deadline_window     = arch.true_deadline_window,
            stated_deadline          = "urgent",                    # always inflated
            true_priority            = arch.true_priority,
            stated_priority          = arch.stated_priority,        # always HIGH or CRITICAL
            true_carbon_flexible     = arch.true_carbon_flexible,
            stated_carbon_flexible   = stated_carbon_flexible,
        )
