"""
Economic layer — Per-window operator observation, team history, oversight flags,
and full-episode ledger.

WindowState is what the scheduler sees at the start of each negotiation window.
EpisodeLedger is the environment's internal accounting — never shown to the LLM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from .job_request import JobRequest


# ── TeamHistory ───────────────────────────────────────────────────────────────

@dataclass
class TeamHistory:
    """
    Per-team behavioural statistics accumulated across windows.
    Visible to the scheduler in the prompt — the primary signal for
    detecting gaming patterns without having access to private fields.

    Updated by ClusterEnvironment after each window resolves, once
    true_priority/true_deadline_window become observable post-hoc
    (i.e. when a job completes or its deadline passes).
    """

    team_id: str

    # ── Submission volume ─────────────────────────────────────────────────────
    total_submitted: int = 0
    total_accepted:  int = 0
    total_deferred:  int = 0
    total_rejected:  int = 0

    # ── Gaming detection rates (updated post-hoc) ─────────────────────────────
    priority_inflation_rate: float = 0.0
    # Fraction of submissions where stated_priority > true_priority.
    # 0.0 = always honest; 0.87 = 87% of claims were inflated.
    # Computed only after jobs complete/expire (true_priority becomes known).

    deadline_compression_rate: float = 0.0
    # Fraction of "urgent" / tight stated_deadlines where the true_deadline_window
    # had slack ≥ 2 windows.

    carbon_gaming_rate: float = 0.0
    # Fraction of stated_carbon_flexible=False claims where true_carbon_flexible=True.

    # ── Completion performance ────────────────────────────────────────────────
    jobs_completed_on_time: int = 0     # completed before true_deadline_window
    jobs_completed_late:    int = 0     # completed after true_deadline_window
    jobs_missed:            int = 0     # deadline passed without execution

    # ── Oversight history ─────────────────────────────────────────────────────
    oversight_flags_received: int = 0
    last_flag_window:         int = -1  # -1 = never flagged

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        if self.total_submitted == 0:
            return 0.0
        return self.total_accepted / self.total_submitted

    @property
    def on_time_rate(self) -> float:
        total_done = self.jobs_completed_on_time + self.jobs_completed_late + self.jobs_missed
        if total_done == 0:
            return 0.0
        return self.jobs_completed_on_time / total_done

    @property
    def is_flagged_as_gaming(self) -> bool:
        """True if any gaming rate exceeds 50% — strong signal."""
        return (
            self.priority_inflation_rate > 0.50
            or self.deadline_compression_rate > 0.50
            or self.carbon_gaming_rate > 0.50
        )

    def summary_str(self) -> str:
        """One-line summary for injection into operator prompt."""
        flag_str = " [GAMING PATTERN DETECTED]" if self.is_flagged_as_gaming else ""
        return (
            f"{self.total_accepted}/{self.total_submitted} accepted | "
            f"priority inflation: {self.priority_inflation_rate:.0%} | "
            f"deadline compression: {self.deadline_compression_rate:.0%} | "
            f"flags: {self.oversight_flags_received}"
            f"{flag_str}"
        )


# ── OversightFlag ─────────────────────────────────────────────────────────────

@dataclass
class OversightFlag:
    """
    A gaming detection signal produced by the OversightMonitor after each window.

    Injected into the NEXT window's WindowState.oversight_flags so the scheduler
    can incorporate it into admission decisions.

    Post-hoc only — the OversightAgent has no real-time veto authority.
    Delayed by one window to maintain clean MDP structure.
    """

    team_id: str

    flag_type: str
    # "priority_inflation"   — stated_priority > true_priority
    # "deadline_compression" — stated urgency inconsistent with true_deadline_window
    # "carbon_gaming"        — stated_carbon_flexible=False when true=True
    # "pattern_gaming"       — persistent multi-window combination of the above

    evidence: str
    # Natural language explanation fed directly into the operator prompt.
    # Should be specific and citable: e.g.
    # "Team B claimed CRITICAL on 5/6 recent jobs; 4 completed 3+ hours after
    #  stated deadline with no consequences — true priority appears LOW/MEDIUM."

    severity: str
    # "warning"   — isolated incident, monitor
    # "flag"      — repeated pattern, recommend conservative admission
    # "escalate"  — systematic gaming, recommend reject until pattern breaks

    confidence: float           # 0.0–1.0
    window_detected: int        # window index when this flag was generated

    def prompt_str(self) -> str:
        """Formatted string for the operator prompt oversight section."""
        return (
            f"[{self.severity.upper()}] {self.team_id}: {self.flag_type} — "
            f"{self.evidence} (confidence: {self.confidence:.0%})"
        )


# ── WindowState ───────────────────────────────────────────────────────────────

@dataclass
class WindowState:
    """
    Complete operator observation for one negotiation window.

    Constructed by ClusterEnvironment at the start of each window and
    serialised into the LLM prompt by training/prompts.py.

    All fields here are PUBLIC (visible to the scheduler).
    Private fields (true_priority, true_deadline_window, etc.) are
    never present in WindowState — they remain in EpisodeLedger only.
    """

    # ── Episode position ──────────────────────────────────────────────────────
    window_idx:    int          # 0–7 (current window)
    total_windows: int = 8
    sim_timestamp: str = "08:00"
    # Simulated wall-clock time at window start.
    # Window 0 = "08:00", window 1 = "09:30", ..., window 7 = "18:30"

    # ── Grid signals ──────────────────────────────────────────────────────────
    carbon_intensity: str = "medium"
    # Current window: "low" | "medium" | "high" | "critical"

    carbon_forecast: list[str] = field(default_factory=list)
    # Intensity for the next 3 windows (window_idx+1 through window_idx+3).
    # e.g. ["high", "high", "low"] — key signal for carbon-aware deferral.
    # Empty list if fewer than 3 windows remain.

    # ── Physical state (coarse summary — operator does not see raw temperatures) ──
    thermal_summary: dict[str, str] = field(default_factory=dict)
    # zone_id → "green" | "yellow" | "red"
    # green:  zone_temp < 23°C  — comfortable, capacity available
    # yellow: 23°C ≤ temp < 25°C — warming, admit cautiously
    # red:    temp ≥ 25°C        — near-limit, avoid adding load to this zone

    capacity_headroom_kw: float = 0.0
    # Total facility power budget minus currently admitted running load (kW).
    # Admitting a job reduces this by job.estimated_kw for job.estimated_duration_hours.

    # ── Job queues ────────────────────────────────────────────────────────────
    pending_requests: list[JobRequest] = field(default_factory=list)
    # New job submissions from teams arriving this window.
    # Scheduler must decide ACCEPT / DEFER / REJECT for each.

    deferred_requests: list[JobRequest] = field(default_factory=list)
    # Jobs from previous windows that were deferred to this window or earlier
    # and have not yet been admitted.
    # Scheduler must decide again — ACCEPT, DEFER further, or REJECT.

    # ── Team context ──────────────────────────────────────────────────────────
    team_history: dict[str, TeamHistory] = field(default_factory=dict)
    # team_id → TeamHistory (accumulated stats visible to scheduler)

    team_budgets_remaining: dict[str, float] = field(default_factory=dict)
    # team_id → GPU-hour equivalents remaining in episode budget.
    # Computed by ChargebackLedger.snapshot().

    # ── Oversight signals (from previous window's oversight run) ──────────────
    oversight_flags: list[OversightFlag] = field(default_factory=list)
    # Flags detected in window (window_idx - 1); empty for window 0.
    # Scheduler should factor these into admission decisions.

    # ── Derived helpers (not serialised to prompt directly) ───────────────────

    @property
    def all_pending(self) -> list[JobRequest]:
        """All jobs requiring a decision: new + previously deferred."""
        return self.pending_requests + self.deferred_requests

    @property
    def total_pending_kw(self) -> float:
        """kW consumed if every pending job were admitted simultaneously."""
        return sum(r.estimated_kw for r in self.all_pending)

    @property
    def oversubscribed_if_all_accepted(self) -> bool:
        """True if accepting all pending jobs would exceed capacity."""
        return self.total_pending_kw > self.capacity_headroom_kw

    def zone_status(self, zone_id: str) -> str:
        return self.thermal_summary.get(zone_id, "unknown")

    def has_red_zones(self) -> bool:
        return any(s == "red" for s in self.thermal_summary.values())

    def is_high_carbon(self) -> bool:
        return self.carbon_intensity in ("high", "critical")

    def low_carbon_window_ahead(self) -> bool:
        """True if any of the next 3 forecast windows is low-carbon."""
        return "low" in self.carbon_forecast

    def next_low_carbon_window(self) -> Optional[int]:
        """Index of the nearest upcoming low-carbon window, or None."""
        for i, intensity in enumerate(self.carbon_forecast):
            if intensity == "low":
                return self.window_idx + 1 + i
        return None

    def team_budget_warning(self, team_id: str) -> bool:
        """True if team has less than 25% of episode budget remaining."""
        remaining = self.team_budgets_remaining.get(team_id, 0.0)
        from .chargeback import TEAM_BUDGET_PER_EPISODE
        return remaining < (TEAM_BUDGET_PER_EPISODE * 0.25)


# ── EpisodeLedger ─────────────────────────────────────────────────────────────

@dataclass
class ActiveJob:
    """
    A job that has been admitted and is currently contributing IT load.
    Held by EpisodeLedger; not visible to the scheduler.
    """
    request:             JobRequest
    admitted_window:     int        # window in which job was admitted
    zone_id:             str        # zone whose IT load this job contributes to
    expected_end_window: int        # window after which the job is considered done
    # Computed as: admitted_window + ceil(estimated_duration_hours / window_duration_h)
    # Default window duration = 1.5 simulated hours (90 min / 60 min per hour)


@dataclass
class CompletedJob:
    """
    Immutable record of a job that has finished execution.
    Used by grader_cluster.py for throughput and carbon reward computation.
    """
    request:                      JobRequest
    admitted_window:              int
    completed_window:             int
    zone_id:                      str
    on_time:                      bool   # completed_window <= request.true_deadline_window
    carbon_intensity_at_run:      str    # grid intensity during the window it ran
    was_deferred_to_low_carbon:   bool
    # True if the job was explicitly deferred (decision=DEFER) AND
    # the window it eventually ran in had carbon_intensity="low".


@dataclass
class EpisodeLedger:
    """
    The environment's full episode accounting.

    Tracks all job state transitions, per-zone IT loads, and
    window-level metrics needed for reward computation.

    NEVER exposed to the LLM or scheduler — internal only.
    """

    # ── Active jobs ───────────────────────────────────────────────────────────
    active_jobs: list[ActiveJob] = field(default_factory=list)
    # Jobs currently running (contributing IT load to their zone).

    # ── Pending deferred jobs ─────────────────────────────────────────────────
    deferred_queue: list[tuple[int, JobRequest]] = field(default_factory=list)
    # (scheduled_window, JobRequest) pairs waiting for their target window.
    # Each window start, ClusterEnvironment moves eligible entries to pending_requests.

    # ── Completed and missed records ──────────────────────────────────────────
    completed_jobs: list[CompletedJob] = field(default_factory=list)
    missed_jobs:    list[JobRequest]   = field(default_factory=list)
    # Missed = deadline passed without the job ever being admitted.

    # ── Window-level metrics (one entry appended per window) ──────────────────
    window_thermal_incidents: list[bool]  = field(default_factory=list)
    # True if any zone exceeded 27°C during that window's physical steps.

    window_rewards:     list[float] = field(default_factory=list)
    window_throughput:  list[float] = field(default_factory=list)
    # jobs_completed_on_time_this_window / jobs_admitted_this_window

    window_carbon_deferrals: list[int] = field(default_factory=list)
    # Count of jobs that ran in a low-carbon window after being explicitly deferred.

    # ── IT load helpers ───────────────────────────────────────────────────────

    def active_load_kw(self, zone_id: str) -> float:
        """Total IT kW from currently running jobs in the given zone."""
        return sum(
            j.request.estimated_kw
            for j in self.active_jobs
            if j.zone_id == zone_id
        )

    def total_active_kw(self) -> float:
        """Total IT kW across all zones from all running jobs."""
        return sum(j.request.estimated_kw for j in self.active_jobs)

    def zone_ids_with_load(self) -> list[str]:
        return list({j.zone_id for j in self.active_jobs})

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def expire_finished_jobs(
        self,
        current_window: int,
        carbon_intensity: str,
    ) -> list[ActiveJob]:
        """
        Remove jobs whose expected_end_window ≤ current_window.
        Appends a CompletedJob record for each expired job.
        Returns the list of expired ActiveJobs (for caller logging).
        """
        still_active, expired = [], []
        for job in self.active_jobs:
            if current_window >= job.expected_end_window:
                expired.append(job)
            else:
                still_active.append(job)
        self.active_jobs = still_active

        for job in expired:
            on_time = job.admitted_window <= job.request.true_deadline_window
            self.completed_jobs.append(CompletedJob(
                request=job.request,
                admitted_window=job.admitted_window,
                completed_window=current_window,
                zone_id=job.zone_id,
                on_time=on_time,
                carbon_intensity_at_run=carbon_intensity,
                was_deferred_to_low_carbon=(
                    job.admitted_window > 0          # was deferred at least once
                    and carbon_intensity == "low"
                    and job.request.true_carbon_flexible
                ),
            ))
        return expired

    def pop_deferred_for_window(self, window_idx: int) -> list[JobRequest]:
        """
        Return all deferred jobs whose scheduled_window == window_idx.
        Removes them from the deferred queue.
        """
        ready, remaining = [], []
        for (scheduled_w, req) in self.deferred_queue:
            if scheduled_w == window_idx:
                ready.append(req)
            else:
                remaining.append((scheduled_w, req))
        self.deferred_queue = remaining
        return ready

    def check_missed_deadlines(self, current_window: int) -> list[JobRequest]:
        """
        Identify deferred jobs whose true_deadline_window has passed.
        Moves them from deferred_queue to missed_jobs. Returns missed list.
        """
        missed, remaining = [], []
        for (scheduled_w, req) in self.deferred_queue:
            if req.true_deadline_window < current_window:
                missed.append(req)
                self.missed_jobs.append(req)
            else:
                remaining.append((scheduled_w, req))
        self.deferred_queue = remaining
        return missed

    # ── Episode-level summary metrics ─────────────────────────────────────────

    def incident_rate(self) -> float:
        """Fraction of windows with at least one thermal incident."""
        if not self.window_thermal_incidents:
            return 0.0
        return sum(self.window_thermal_incidents) / len(self.window_thermal_incidents)

    def overall_throughput(self) -> float:
        """Episode-level fraction of jobs completed before true deadline."""
        total = len(self.completed_jobs) + len(self.missed_jobs)
        if total == 0:
            return 0.0
        on_time = sum(1 for j in self.completed_jobs if j.on_time)
        return on_time / total

    def carbon_deferral_rate(self) -> float:
        """
        Fraction of carbon-flexible jobs that ran in a low-carbon window
        after being explicitly deferred.
        """
        eligible = [j for j in self.completed_jobs if j.request.true_carbon_flexible]
        if not eligible:
            return 0.0
        deferred_to_low = sum(1 for j in eligible if j.was_deferred_to_low_carbon)
        return deferred_to_low / len(eligible)

    def compute_window_duration_windows(
        self, duration_hours: float, window_duration_hours: float = 1.5
    ) -> int:
        """Convert job duration in hours to number of windows it occupies."""
        return max(1, math.ceil(duration_hours / window_duration_hours))
