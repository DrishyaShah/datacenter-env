"""
Economic layer — Job request and admission decision dataclasses.

These are the LOCKED schemas for the ClusterEnv negotiation protocol.
Changing any field name or type here will break prompt formatting,
reward computation, and oversight detection simultaneously.
Do not modify without updating prompts.py, grader_cluster.py, and scripted_teams.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Priority cost multipliers ─────────────────────────────────────────────────
# Shared by JobRequest.__post_init__ and ChargebackLedger.
# Higher stated priority → higher budget cost → natural disincentive against
# inflation for teams that actually care about their budget.

PRIORITY_COST_MULTIPLIERS: dict[str, float] = {
    "LOW":      1.0,
    "MEDIUM":   1.5,
    "HIGH":     2.0,
    "CRITICAL": 3.0,
}

# Validation sets — used by parsers and test assertions
VALID_STATED_PRIORITIES: frozenset[str] = frozenset(PRIORITY_COST_MULTIPLIERS.keys())
VALID_TRUE_PRIORITIES:   frozenset[str] = frozenset(PRIORITY_COST_MULTIPLIERS.keys())
VALID_DECISIONS:         frozenset[str] = frozenset({"ACCEPT", "DEFER", "REJECT"})

# Priority ordering for inflation detection (LOW=0 → CRITICAL=3)
PRIORITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ── JobRequest ────────────────────────────────────────────────────────────────

@dataclass
class JobRequest:
    """
    A compute job submitted by a team in one negotiation window.

    INFORMATION ASYMMETRY — two categories of fields:

    PUBLIC (visible to scheduler in the prompt):
        request_id, team_id, job_type, job_description,
        estimated_kw, estimated_duration_hours,
        stated_deadline, stated_priority, stated_carbon_flexible,
        compute_budget_cost

    PRIVATE (held by environment; used only for reward computation and
             oversight detection — never passed to the LLM):
        true_deadline_window, true_priority, true_carbon_flexible

    In v1, kW and duration are accurate (no power-draw gaming).
    Gaming is deadline/priority/carbon misrepresentation only.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    request_id: str
    # Canonical format: "req_{window_idx}_{team_id}_{idx}"
    # e.g. "req_3_team_b_0" = window 3, team_b, first request this window

    team_id: str
    # "team_a" (cooperative) | "team_b" (strategic)

    # ── Semantic content — LLM must parse and reason about this ──────────────
    job_type: str
    # "training" | "inference" | "batch" | "fine_tuning" | "evaluation"

    job_description: str
    # Free-text natural language description.
    # The scheduler must infer true workload characteristics from this.
    # Team B uses plausible-sounding but misleading descriptions.

    # ── Resource requirements (accurate in v1) ────────────────────────────────
    estimated_kw: float
    # Stated power draw in kW. Accurate in v1 — kW gaming is not modelled.
    # Range: 50–500 kW depending on job type.

    estimated_duration_hours: float
    # Stated job duration in hours. Accurate in v1.
    # Range: 1–12 hours.

    # ── Deadline information (ASYMMETRIC) ─────────────────────────────────────
    true_deadline_window: int
    # PRIVATE. The actual hard deadline window index [0–7].
    # A job not started by this window is considered missed.
    # Never revealed to the scheduler — only used post-hoc for reward.

    stated_deadline: str
    # PUBLIC. What the team chooses to reveal.
    # Team A: accurate reflection of true_deadline_window.
    # Team B: always states "urgent" or "by window N" (N underestimates slack).
    # Values: "urgent" | "flexible" | "by window N" | "end of day" | "no deadline"

    # ── Priority (ASYMMETRIC — Team B always inflates) ────────────────────────
    true_priority: str
    # PRIVATE. Actual priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    # Reflects genuine business impact of the job.

    stated_priority: str
    # PUBLIC. What the team declares.
    # Team A: matches true_priority.
    # Team B: always states "HIGH" or "CRITICAL" regardless of true_priority.
    # Values: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"

    # ── Carbon flexibility (ASYMMETRIC — Team B misrepresents) ───────────────
    true_carbon_flexible: bool
    # PRIVATE. Can this job safely be deferred to a low-carbon window?
    # Used to compute carbon deferral reward and detect carbon_gaming.

    stated_carbon_flexible: bool
    # PUBLIC. What the team declares.
    # Team A: matches true_carbon_flexible.
    # Team B: sometimes states False when true value is True,
    #          to appear urgent and avoid being deferred.

    # ── Chargeback cost (auto-computed from stated_priority + duration) ───────
    compute_budget_cost: float = field(init=False)
    # GPU-hour equivalents consumed from team's episode budget if ACCEPTED.
    # Formula: estimated_duration_hours × PRIORITY_COST_MULTIPLIERS[stated_priority]
    # This creates a natural penalty for priority inflation:
    # claiming CRITICAL costs 3× LOW from the team's finite budget.

    def __post_init__(self) -> None:
        multiplier = PRIORITY_COST_MULTIPLIERS.get(self.stated_priority, 1.0)
        self.compute_budget_cost = round(self.estimated_duration_hours * multiplier, 2)

    # ── Derived properties ────────────────────────────────────────────────────

    def is_gaming_priority(self) -> bool:
        """True if team stated a higher priority than the true priority."""
        return (
            PRIORITY_ORDER.get(self.stated_priority, 0)
            > PRIORITY_ORDER.get(self.true_priority, 0)
        )

    def is_gaming_deadline(self, current_window: int) -> bool:
        """True if team claimed urgency but has ≥2 windows of actual slack."""
        urgent_claims = {"urgent", "by window 0", "by window 1", "by window 2"}
        stated_is_urgent = (
            self.stated_deadline == "urgent"
            or self.stated_deadline.startswith("by window")
        )
        actual_slack = self.true_deadline_window - current_window
        return stated_is_urgent and actual_slack >= 2

    def is_gaming_carbon(self) -> bool:
        """True if team claimed carbon-inflexible but true value is flexible."""
        return self.true_carbon_flexible and not self.stated_carbon_flexible

    def deadline_slack(self, current_window: int) -> int:
        """Windows remaining before true hard deadline."""
        return max(0, self.true_deadline_window - current_window)

    def is_genuinely_urgent(self, current_window: int) -> bool:
        """True deadline within 1 window — must be scheduled now or missed."""
        return self.deadline_slack(current_window) <= 1

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def public_fields(self) -> dict:
        """
        Fields visible to the scheduler agent.
        Excludes all private fields (true_deadline_window, true_priority,
        true_carbon_flexible). Safe to pass into prompt templates.
        """
        return {
            "request_id":               self.request_id,
            "team_id":                  self.team_id,
            "job_type":                 self.job_type,
            "job_description":          self.job_description,
            "estimated_kw":             self.estimated_kw,
            "estimated_duration_hours": self.estimated_duration_hours,
            "stated_deadline":          self.stated_deadline,
            "stated_priority":          self.stated_priority,
            "stated_carbon_flexible":   self.stated_carbon_flexible,
            "compute_budget_cost":      self.compute_budget_cost,
        }

    def ground_truth_fields(self) -> dict:
        """
        All fields including private ones.
        Used only by OversightMonitor and reward computation — never the LLM.
        """
        return {
            **self.public_fields(),
            "true_deadline_window":  self.true_deadline_window,
            "true_priority":         self.true_priority,
            "true_carbon_flexible":  self.true_carbon_flexible,
            "is_gaming_priority":    self.is_gaming_priority(),
            "is_gaming_carbon":      self.is_gaming_carbon(),
        }


# ── AdmissionDecision ─────────────────────────────────────────────────────────

@dataclass
class AdmissionDecision:
    """
    The scheduler's decision for a single job request in one negotiation window.

    Produced by parsing the LLM's JSON output. The ClusterEnvironment
    validates each decision and applies it to the EpisodeLedger.
    """

    request_id: str
    # Must match an existing JobRequest.request_id in the current window.

    decision: str
    # "ACCEPT"  — admit now; job starts this window, kW added to zone IT load
    # "DEFER"   — schedule for a future window; must specify scheduled_window
    # "REJECT"  — permanently decline; job is dropped, team notified

    scheduled_window: Optional[int]
    # Required when decision == "DEFER".
    # Target window index [0–7]; must be strictly > current window_idx.
    # The environment will attempt to start the job in that window if capacity allows.
    # None for ACCEPT and REJECT.

    reasoning: str
    # LLM's one-sentence explanation for this specific decision.
    # Stored in episode logs; not graded in v1.
    # Used for demo narration and oversight cross-reference.

    def __post_init__(self) -> None:
        if self.decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision '{self.decision}'. "
                f"Must be one of {sorted(VALID_DECISIONS)}."
            )
        if self.decision == "DEFER" and self.scheduled_window is None:
            raise ValueError(
                "DEFER decision requires scheduled_window to be set (int [0–7])."
            )
        if self.decision != "DEFER" and self.scheduled_window is not None:
            # Normalise: non-DEFER decisions should not carry scheduled_window
            self.scheduled_window = None

    # ── Factory constructors (used by rule-based baseline scheduler) ──────────

    @classmethod
    def accept(cls, request_id: str, reasoning: str = "accepted") -> "AdmissionDecision":
        return cls(request_id=request_id, decision="ACCEPT",
                   scheduled_window=None, reasoning=reasoning)

    @classmethod
    def defer(cls, request_id: str, target_window: int,
              reasoning: str = "deferred") -> "AdmissionDecision":
        return cls(request_id=request_id, decision="DEFER",
                   scheduled_window=target_window, reasoning=reasoning)

    @classmethod
    def reject(cls, request_id: str, reasoning: str = "rejected") -> "AdmissionDecision":
        return cls(request_id=request_id, decision="REJECT",
                   scheduled_window=None, reasoning=reasoning)

    def to_dict(self) -> dict:
        return {
            "request_id":       self.request_id,
            "decision":         self.decision,
            "scheduled_window": self.scheduled_window,
            "reasoning":        self.reasoning,
        }
