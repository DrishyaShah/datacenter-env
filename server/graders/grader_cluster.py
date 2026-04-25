"""
Reward grader for ClusterEnv — 3-component window-level reward.

Operates at the negotiation-window level (8 windows per episode), not the
physical step level. ClusterEnvironment calls record_window() once after
each window's 18 physical steps complete, then final_score() at episode end.

Reward formula (locked):
    R_window = 0.50 × R_throughput  +  0.35 × R_thermal  +  0.15 × R_carbon

    R_throughput = jobs_completed_on_time / max(jobs_admitted, 1)
    R_thermal    = -1.0 if power budget was violated at any physical step, else 0.0
    R_carbon     = carbon_deferred_completions / max(carbon_flexible_admitted, 1)

    Final score  = mean(R_window) across all 8 windows ∈ [-0.35, +0.65]

Incident metric: power_budget_violated() from cluster_scenario.py
    (total_it_load_kw > TOTAL_POWER_BUDGET_KW = 900 kW)
    Temperature-based incidents are not used — the CoolingHeuristic prevents
    zone temperature breaches at realistic job sizes; only the scheduler can
    prevent power budget violations through admission control.

The grader is stateless with respect to EpisodeLedger and FacilityState.
ClusterEnvironment computes the four scalar counts and passes them here.
This keeps the grader independently testable and decoupled from physics changes.

Gate verification:
    All-incident window  → R_window = -0.35
    Perfect window       → R_window = +0.65
"""

from __future__ import annotations


class ClusterGrader:
    """
    Window-level reward accumulator for one ClusterEnv episode.

    Usage:
        grader = ClusterGrader()
        for window_idx in range(8):
            reward = grader.record_window(
                window_idx=window_idx,
                jobs_admitted=n_admitted,
                jobs_completed_on_time=n_on_time,
                power_violated=any_step_violated,
                carbon_flexible_admitted=n_flex_admitted,
                carbon_deferred_completions=n_deferred_to_low_carbon,
            )
        score = grader.final_score()
        metrics = grader.component_means()   # → W&B
    """

    # ── Reward weights (locked) ───────────────────────────────────────────────
    W_THROUGHPUT: float = 0.50
    W_THERMAL:    float = 0.35   # multiplied by R_thermal ∈ {-1.0, 0.0}
    W_CARBON:     float = 0.15

    # ── Baseline reference values (priority_weighted_threshold scheduler) ─────
    # Measured over 10 episodes (tests/test_cluster.py calibration gate).
    # priority_weighted_threshold is safe (0% incidents) but suboptimal on
    # throughput and carbon. Trained agent must beat these scores.
    # Note: accept_all has 92% incident rate — that's the "naive" upper bound.
    BASELINE_INCIDENT_RATE:   float = 0.00   # rule-based avoids violations; LLM target <15%
    BASELINE_THROUGHPUT:      float = 0.54
    BASELINE_CARBON_DEFERRAL: float = 0.06

    # ── Training targets (trained scheduler goals) ────────────────────────────
    TARGET_INCIDENT_RATE:   float = 0.15
    TARGET_CARBON_DEFERRAL: float = 0.40

    def __init__(self) -> None:
        self._records: list[dict] = []

    # ── Core interface ────────────────────────────────────────────────────────

    def record_window(
        self,
        window_idx: int,
        jobs_admitted: int,
        jobs_completed_on_time: int,
        power_violated: bool,
        carbon_flexible_admitted: int,
        carbon_deferred_completions: int,
    ) -> float:
        """
        Compute R_window for one negotiation window and append to internal log.

        Parameters
        ----------
        window_idx : int
            Window index [0–7]. Stored in the record for logging; not used in math.
        jobs_admitted : int
            Jobs accepted (decision=ACCEPT) during this window, including those
            promoted from the deferred queue.
        jobs_completed_on_time : int
            Jobs whose expected_end_window fell within this window AND whose
            admitted_window ≤ true_deadline_window (on-time by true deadline).
        power_violated : bool
            True if power_budget_violated() returned True at any of the 18
            physical steps in this window.
        carbon_flexible_admitted : int
            Count of admitted jobs with true_carbon_flexible=True. Denominator
            for R_carbon — represents the pool of jobs the scheduler could have
            shifted to a low-carbon window.
        carbon_deferred_completions : int
            Count of jobs completing this window where was_deferred_to_low_carbon=True:
            job was explicitly deferred at least once AND ran in a low-carbon window.

        Returns
        -------
        float
            R_window ∈ [-0.35, +0.65].
        """
        r_throughput = jobs_completed_on_time / max(jobs_admitted, 1)
        r_thermal    = -1.0 if power_violated else 0.0
        r_carbon     = carbon_deferred_completions / max(carbon_flexible_admitted, 1)

        reward = (
            self.W_THROUGHPUT * r_throughput
            + self.W_THERMAL  * r_thermal
            + self.W_CARBON   * r_carbon
        )

        self._records.append({
            "window_idx":                  window_idx,
            "jobs_admitted":               jobs_admitted,
            "jobs_completed_on_time":      jobs_completed_on_time,
            "power_violated":              power_violated,
            "carbon_flexible_admitted":    carbon_flexible_admitted,
            "carbon_deferred_completions": carbon_deferred_completions,
            "r_throughput":                r_throughput,
            "r_thermal":                   r_thermal,
            "r_carbon":                    r_carbon,
            "reward":                      reward,
        })

        return reward

    def final_score(self) -> float:
        """
        Episode-level score: mean of all per-window R_window values.
        Call after the last window's record_window() completes.
        Returns 0.0 for an empty episode (no windows recorded).
        """
        if not self._records:
            return 0.0
        return sum(r["reward"] for r in self._records) / len(self._records)

    # ── Episode-level metric helpers (W&B logging) ───────────────────────────

    def incident_rate(self) -> float:
        """Fraction of recorded windows where power budget was violated."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r["power_violated"]) / len(self._records)

    def mean_throughput(self) -> float:
        """Mean R_throughput across all recorded windows."""
        if not self._records:
            return 0.0
        return sum(r["r_throughput"] for r in self._records) / len(self._records)

    def mean_carbon_deferral(self) -> float:
        """Mean R_carbon across all recorded windows."""
        if not self._records:
            return 0.0
        return sum(r["r_carbon"] for r in self._records) / len(self._records)

    def component_means(self) -> dict:
        """
        All four training metrics in one dict — pass directly to wandb.log().

        Keys match the W&B metric names from train_grpo.py:
            reward/mean, metrics/incident_rate,
            reward/throughput_component, reward/carbon_component
        """
        return {
            "reward/mean":               self.final_score(),
            "reward/throughput_component": self.mean_throughput(),
            "reward/carbon_component":   self.mean_carbon_deferral(),
            "metrics/incident_rate":     self.incident_rate(),
        }

    def window_log(self) -> list[dict]:
        """Full per-window records for episode debugging and demo replay."""
        return list(self._records)
