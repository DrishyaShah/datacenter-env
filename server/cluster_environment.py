"""
ClusterEnvironment -- 8-window AI cluster scheduling environment.

Episode structure: 8 negotiation windows x 18 physical steps = 144 total steps.
Each window: scheduler issues admission decisions -> cooling runs 18 physics steps ->
window metrics recorded -> next window observation returned.

Interface:
    env = ClusterEnvironment(cooling_controller=None, enable_chiller_fault=True)
    window_state = env.reset(seed=42)
    window_state, reward, done, info = env.step(decisions)

cooling_controller defaults to CoolingHeuristic (rule-based). Pass a trained PPO
agent for two-phase training (Person B's PPO controller).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from server.agents.cooling_heuristic import CoolingHeuristic
from server.agents.oversight_monitor import OversightMonitor
from server.agents.scripted_teams import CooperativeTeam, StrategicTeam
from server.economic import (
    WindowState,
    EpisodeLedger,
    TeamHistory,
    ActiveJob,
)
from server.economic.chargeback import ChargebackLedger
from server.economic.job_request import AdmissionDecision, JobRequest, PRIORITY_ORDER
from server.economic.window_state import OversightFlag
from server.graders.grader_cluster import ClusterGrader
from server.scenarios.cluster_scenario import (
    CARBON_SCHEDULE,
    CARBON_NUMERIC_SCHEDULE,
    OUTSIDE_TEMP_SCHEDULE,
    WET_BULB_SCHEDULE,
    WINDOWS_PER_EPISODE,
    PHYSICAL_STEPS_PER_WINDOW,
    assign_zone,
    build_cluster_facility,
    compute_headroom_kw,
    power_budget_violated,
    thermal_summary,
    window_to_hour,
    window_to_timestamp,
)
from server.simulation import FacilityState


class ClusterEnvironment:
    """
    Gym-style environment for the ClusterEnv scheduling task.

    reset() -> WindowState (window 0 observation)
    step(decisions) -> (WindowState, reward, done, info)

    cooling_controller: any object implementing
        step(facility, upcoming_load_kw=None) -> _DCActionStub
        initial_action(zones) -> _DCActionStub   (static / classmethod)
    Defaults to CoolingHeuristic().
    """

    def __init__(
        self,
        cooling_controller=None,
        enable_chiller_fault: bool = True,
    ) -> None:
        self.cooling_controller = cooling_controller or CoolingHeuristic()
        self.enable_chiller_fault = enable_chiller_fault

        # Episode objects (None until reset())
        self._facility:   Optional[FacilityState]    = None
        self._ledger:     Optional[EpisodeLedger]    = None
        self._chargeback: Optional[ChargebackLedger] = None
        self._grader:     Optional[ClusterGrader]    = None
        self._window_idx: int                        = 0
        self._done:       bool                       = False
        self._last_action                            = None
        self._rng:        Optional[np.random.Generator] = None

        # Current window's job queues (rebuilt each window)
        self._pending_requests: list[JobRequest] = []
        self._deferred_display: list[JobRequest] = []
        self._request_index:    dict[str, JobRequest] = {}

        # Team agents (stateless; reused across episodes)
        self._team_a = CooperativeTeam("team_a")
        self._team_b = StrategicTeam("team_b")

        # TeamHistory updated incrementally throughout episode
        self._team_history: dict[str, TeamHistory] = {}

        self._pending_flags: list[OversightFlag] = []
        self._oversight_monitor: OversightMonitor = OversightMonitor()

    # -- Public interface ------------------------------------------------------

    def reset(self, seed: int | None = None) -> WindowState:
        """Reset episode. Returns window-0 WindowState."""
        _seed = seed if seed is not None else int(np.random.randint(0, 99_999))
        self._rng = np.random.default_rng(_seed)

        self._facility = build_cluster_facility(
            seed=_seed, window_idx=0, enable_chiller_fault=self.enable_chiller_fault
        )
        self._ledger     = EpisodeLedger()
        self._chargeback = ChargebackLedger()
        self._chargeback.register_team("team_a")
        self._chargeback.register_team("team_b")
        self._grader     = ClusterGrader()
        self._window_idx = 0
        self._done       = False

        self._team_history = {
            "team_a": TeamHistory(team_id="team_a"),
            "team_b": TeamHistory(team_id="team_b"),
        }
        self._pending_flags = []
        self._oversight_monitor = OversightMonitor()

        self._last_action = CoolingHeuristic.initial_action(self._facility.zones)

        carbon_0 = CARBON_SCHEDULE[0]
        self._pending_requests = self._generate_window_requests(0, carbon_0)
        self._deferred_display = []
        self._request_index    = {r.request_id: r for r in self._pending_requests}

        return self._build_window_state()

    def step(
        self, decisions: list[AdmissionDecision]
    ) -> tuple[WindowState, float, bool, dict]:
        """
        Process admission decisions for current window, run 18 physical steps,
        advance to next window.

        Returns (next_window_state, reward, done, info).
        On done=True, next_window_state is a terminal placeholder.
        """
        if self._facility is None:
            raise RuntimeError("Call reset() before step().")

        # -- Phase 1: Admission control ----------------------------------------
        jobs_admitted            = 0
        carbon_flexible_admitted = 0

        for dec in decisions:
            req = self._request_index.get(dec.request_id)
            if req is None:
                continue  # unknown id -- LLM hallucinated; skip silently

            if dec.decision == "ACCEPT":
                if not self._chargeback.can_afford(req):
                    # Budget exhausted: force defer rather than silently drop
                    target = self._window_idx + 1
                    if target < WINDOWS_PER_EPISODE:
                        self._ledger.deferred_queue.append((target, req))
                    self._team_history[req.team_id].total_deferred += 1
                    continue

                self._chargeback.charge(req)
                zone_id    = assign_zone(req.team_id, self._facility)
                duration_w = self._ledger.compute_window_duration_windows(
                    req.estimated_duration_hours
                )
                self._ledger.active_jobs.append(ActiveJob(
                    request=req,
                    admitted_window=self._window_idx,
                    zone_id=zone_id,
                    expected_end_window=self._window_idx + duration_w,
                ))
                jobs_admitted += 1
                self._team_history[req.team_id].total_accepted += 1
                if req.true_carbon_flexible:
                    carbon_flexible_admitted += 1

            elif dec.decision == "DEFER":
                target = int(dec.scheduled_window) if dec.scheduled_window is not None \
                         else self._window_idx + 1
                target = max(target, self._window_idx + 1)
                target = min(target, WINDOWS_PER_EPISODE - 1)
                self._ledger.deferred_queue.append((target, req))
                self._team_history[req.team_id].total_deferred += 1

            else:  # REJECT
                self._team_history[req.team_id].total_rejected += 1

        # -- Phase 2: Physical simulation --------------------------------------
        load_map: dict[str, float] = {}
        for job in self._ledger.active_jobs:
            load_map[job.zone_id] = (
                load_map.get(job.zone_id, 0.0) + job.request.estimated_kw
            )
        self._facility.set_all_job_loads(load_map)

        power_violated = False
        upcoming       = self._upcoming_load_kw()

        for _ in range(PHYSICAL_STEPS_PER_WINDOW):
            action = self.cooling_controller.step(
                self._facility, upcoming_load_kw=upcoming
            )
            self._facility.step(action, self._last_action)
            self._last_action = action
            if power_budget_violated(self._facility):
                power_violated = True

        # -- Phase 3: Window completion metrics --------------------------------
        carbon = CARBON_SCHEDULE[self._window_idx]
        self._ledger.expire_finished_jobs(self._window_idx, carbon)

        new_completions = [
            j for j in self._ledger.completed_jobs
            if j.completed_window == self._window_idx
        ]
        jobs_completed_on_time      = sum(1 for j in new_completions if j.on_time)
        carbon_deferred_completions = sum(
            1 for j in new_completions if j.was_deferred_to_low_carbon
        )

        for comp in new_completions:
            th = self._team_history[comp.request.team_id]
            if comp.on_time:
                th.jobs_completed_on_time += 1
            else:
                th.jobs_completed_late += 1

        reward = self._grader.record_window(
            window_idx=self._window_idx,
            jobs_admitted=jobs_admitted,
            jobs_completed_on_time=jobs_completed_on_time,
            power_violated=power_violated,
            carbon_flexible_admitted=carbon_flexible_admitted,
            carbon_deferred_completions=carbon_deferred_completions,
        )

        # Oversight: analyze current window's requests for gaming patterns.
        # _pending_requests / _deferred_display still hold this window's jobs
        # (Phase 4 hasn't run yet), so pass them directly.
        current_requests = list(self._pending_requests) + list(self._deferred_display)
        self._pending_flags = self._oversight_monitor.analyze_window(
            window_idx=self._window_idx,
            requests=current_requests,
            decisions=decisions,
            team_histories=self._team_history,
        )
        for flag in self._pending_flags:
            th = self._team_history.get(flag.team_id)
            if th:
                th.oversight_flags_received += 1
                th.last_flag_window = self._window_idx

        # -- Phase 4: Advance window -------------------------------------------
        self._window_idx += 1
        done = self._window_idx >= WINDOWS_PER_EPISODE

        if not done:
            self._update_weather(self._window_idx)
            self._last_action = CoolingHeuristic.initial_action(self._facility.zones)

            missed = self._ledger.check_missed_deadlines(self._window_idx)
            for req in missed:
                self._team_history[req.team_id].jobs_missed += 1

            deferred_now = self._ledger.pop_deferred_for_window(self._window_idx)

            new_carbon    = CARBON_SCHEDULE[self._window_idx]
            new_requests  = self._generate_window_requests(self._window_idx, new_carbon)

            self._pending_requests = new_requests
            self._deferred_display = deferred_now
            self._request_index    = {
                r.request_id: r for r in new_requests + deferred_now
            }

        info: dict = {
            "window_idx":             self._window_idx - 1,
            "reward":                 reward,
            "power_violated":         power_violated,
            "jobs_admitted":          jobs_admitted,
            "jobs_completed_on_time": jobs_completed_on_time,
        }
        if done:
            info.update(self._grader.component_means())

        return self._build_window_state(), reward, done, info

    # -- Private helpers -------------------------------------------------------

    def _generate_window_requests(
        self, window_idx: int, carbon: str
    ) -> list[JobRequest]:
        """Generate requests from both teams; update TeamHistory gaming rates."""
        reqs_a    = self._team_a.generate_window_requests(window_idx, carbon, self._rng)
        reqs_b    = self._team_b.generate_window_requests(window_idx, carbon, self._rng)
        all_reqs  = reqs_a + reqs_b

        for req in all_reqs:
            th  = self._team_history[req.team_id]
            sub = th.total_submitted + 1
            th.total_submitted = sub

            if PRIORITY_ORDER.get(req.stated_priority, 0) > PRIORITY_ORDER.get(
                req.true_priority, 0
            ):
                th.priority_inflation_rate = (
                    th.priority_inflation_rate * (sub - 1) + 1.0
                ) / sub

            if req.stated_deadline == "urgent" and req.true_deadline_window > window_idx + 1:
                th.deadline_compression_rate = (
                    th.deadline_compression_rate * (sub - 1) + 1.0
                ) / sub

            if req.true_carbon_flexible and not req.stated_carbon_flexible:
                th.carbon_gaming_rate = (
                    th.carbon_gaming_rate * (sub - 1) + 1.0
                ) / sub

        return all_reqs

    def _update_weather(self, window_idx: int) -> None:
        """Update facility weather fields between windows (thermal state preserved)."""
        self._facility.outside_temp_c                  = OUTSIDE_TEMP_SCHEDULE[window_idx]
        self._facility.wet_bulb_temp_c                 = WET_BULB_SCHEDULE[window_idx]
        self._facility.grid_carbon_intensity            = CARBON_SCHEDULE[window_idx]
        self._facility.grid_carbon_intensity_normalized = CARBON_NUMERIC_SCHEDULE[window_idx]
        self._facility.timestamp_hour                   = window_to_hour(window_idx)

    def _build_window_state(self) -> WindowState:
        """Assemble WindowState from current episode state for LLM observation."""
        w = self._window_idx
        if w >= WINDOWS_PER_EPISODE:
            return WindowState(
                window_idx=w,
                total_windows=WINDOWS_PER_EPISODE,
                sim_timestamp=window_to_timestamp(WINDOWS_PER_EPISODE - 1),
            )
        return WindowState(
            window_idx=w,
            total_windows=WINDOWS_PER_EPISODE,
            sim_timestamp=window_to_timestamp(w),
            carbon_intensity=CARBON_SCHEDULE[w],
            carbon_forecast=CARBON_SCHEDULE[w + 1: w + 4],
            thermal_summary=thermal_summary(self._facility),
            capacity_headroom_kw=compute_headroom_kw(self._facility),
            pending_requests=list(self._pending_requests),
            deferred_requests=list(self._deferred_display),
            team_history=dict(self._team_history),
            team_budgets_remaining=self._chargeback.snapshot(),
            oversight_flags=list(self._pending_flags),
        )

    def _upcoming_load_kw(self, steps_ahead: int = 3) -> list[float]:
        """IT load forecast for next N physical steps (constant within a window)."""
        load = self._ledger.total_active_kw()
        return [load] * steps_ahead
