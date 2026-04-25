"""
Economic layer -- Internal compute chargeback ledger.

Tracks per-team budget consumption across the episode.
Priority-weighted pricing: higher stated_priority -> higher cost from budget.

Why this exists:
  Without a budget constraint, an agent could game the scheduler indefinitely
  by always claiming CRITICAL. The chargeback ledger creates a finite resource
  that makes priority inflation self-limiting for rational teams: claiming CRITICAL
  costs 3x LOW from the team's episode budget, so a team that always inflates
  exhausts their budget before their legitimate jobs can run.

  This is also what makes the scheduler's decisions semantic rather than purely
  numeric: it must reason about whether a HIGH claim is worth the 2x budget cost
  given the team's history and remaining balance.
"""

from __future__ import annotations
from .job_request import JobRequest, PRIORITY_COST_MULTIPLIERS


# -- Budget constants ----------------------------------------------------------

TEAM_BUDGET_PER_EPISODE: float = 200.0
# GPU-hour equivalents per team per 8-window episode.
#
# Calibration:
#   Average job: 4 hours x 1.5 avg multiplier = 6 units per job.
#   Budget of 200 -> ~33 jobs at average priority.
#   Each team submits 1-2 jobs per window x 8 windows = 8-16 jobs.
#   Budget is intentionally ~2x expected demand so honest teams never run out,
#   while a team claiming CRITICAL on everything (3x multiplier) consumes
#   ~4-6h x 3.0 = 12-18 units/job -> 200 / 15  13 jobs max.
#
# This means the strategic team (Team B) that always claims CRITICAL will
# exhaust their budget by window 5-6, creating a visible signal for the scheduler.

BUDGET_WARNING_THRESHOLD: float = 0.25
# Fraction of budget remaining at which a warning appears in the operator prompt.
# At 25% remaining, the team has limited capacity for high-priority claims.

BUDGET_CRITICAL_THRESHOLD: float = 0.10
# Fraction at which the operator prompt shows a critical budget alert.


class ChargebackLedger:
    """
    Tracks compute budget consumption per team across the full episode.

    The pricing mechanism:
    - Each team starts with TEAM_BUDGET_PER_EPISODE GPU-hour equivalents.
    - Admitting a job charges: estimated_duration_hours x priority_multiplier
    - Budget is deducted at admission time (ACCEPT decision).
    - Rejected/deferred jobs are not charged.
    - Budget is not refunded if a job is preempted (not in v1 scope).

    The scheduler sees remaining balances via WindowState.team_budgets_remaining.
    A team nearing zero budget is a strong signal that it has been claiming
    high priority excessively -- exactly what the scheduler should learn to detect.
    """

    def __init__(self) -> None:
        self._budgets:         dict[str, float] = {}
        self._spend:           dict[str, float] = {}
        self._transaction_log: list[dict]       = []

    # -- Setup -----------------------------------------------------------------

    def register_team(
        self,
        team_id: str,
        budget: float = TEAM_BUDGET_PER_EPISODE,
    ) -> None:
        """Register a team with a fresh budget. Call once per episode."""
        self._budgets[team_id] = budget
        self._spend[team_id]   = 0.0

    # -- Core operations -------------------------------------------------------

    def compute_cost(self, request: JobRequest) -> float:
        """
        Cost in GPU-hour equivalents for admitting this request.
        Uses stated_priority -- the team's declared (potentially inflated) priority.
        """
        multiplier = PRIORITY_COST_MULTIPLIERS.get(request.stated_priority, 1.0)
        return round(request.estimated_duration_hours * multiplier, 3)

    def can_afford(self, request: JobRequest) -> bool:
        """True if the team has sufficient budget to cover this job's cost."""
        return self.remaining(request.team_id) >= self.compute_cost(request)

    def charge(self, request: JobRequest) -> float:
        """
        Deduct admission cost from the team's budget.
        Call when decision == ACCEPT.
        Returns the cost charged.
        """
        cost = self.compute_cost(request)
        self._spend[request.team_id] = (
            self._spend.get(request.team_id, 0.0) + cost
        )
        self._transaction_log.append({
            "request_id":      request.request_id,
            "team_id":         request.team_id,
            "stated_priority": request.stated_priority,
            "duration_hours":  request.estimated_duration_hours,
            "cost":            cost,
            "budget_after":    self.remaining(request.team_id),
        })
        return cost

    # -- Query helpers ---------------------------------------------------------

    def remaining(self, team_id: str) -> float:
        """Budget remaining for this team (GPU-hour equivalents)."""
        return max(
            0.0,
            self._budgets.get(team_id, 0.0) - self._spend.get(team_id, 0.0),
        )

    def fraction_remaining(self, team_id: str) -> float:
        """Fraction of episode budget remaining [0.0-1.0]."""
        budget = self._budgets.get(team_id, 1.0)
        return self.remaining(team_id) / budget if budget > 0 else 0.0

    def total_spend(self, team_id: str) -> float:
        return self._spend.get(team_id, 0.0)

    def is_budget_warning(self, team_id: str) -> bool:
        """True if remaining budget  BUDGET_WARNING_THRESHOLD."""
        return self.fraction_remaining(team_id) <= BUDGET_WARNING_THRESHOLD

    def is_budget_critical(self, team_id: str) -> bool:
        """True if remaining budget  BUDGET_CRITICAL_THRESHOLD."""
        return self.fraction_remaining(team_id) <= BUDGET_CRITICAL_THRESHOLD

    def snapshot(self) -> dict[str, float]:
        """
        Current remaining budget per team.
        Injected into WindowState.team_budgets_remaining every window.
        """
        return {tid: self.remaining(tid) for tid in self._budgets}

    # -- Prompt helpers --------------------------------------------------------

    def summary_str(self, team_id: str) -> str:
        """
        One-line budget summary for the operator prompt.
        e.g. "142.5/200 GPU-hr equiv (71% remaining)"
              "32.0/200 GPU-hr equiv (16% remaining)  LOW"
        """
        remaining = self.remaining(team_id)
        total     = self._budgets.get(team_id, TEAM_BUDGET_PER_EPISODE)
        pct       = self.fraction_remaining(team_id)

        if self.is_budget_critical(team_id):
            tag = "  CRITICAL"
        elif self.is_budget_warning(team_id):
            tag = "  LOW"
        else:
            tag = ""

        return f"{remaining:.1f}/{total:.0f} GPU-hr equiv ({pct:.0%} remaining){tag}"

    # -- Audit -----------------------------------------------------------------

    def transaction_log(self) -> list[dict]:
        """Full transaction history -- for episode debugging and oversight."""
        return list(self._transaction_log)

    def team_ids(self) -> list[str]:
        return list(self._budgets.keys())
