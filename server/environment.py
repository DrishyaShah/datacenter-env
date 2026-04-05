"""
DC-OpenEnv: Data Centre Operations Environment — V2

Fully OpenEnv-compliant environment.
Manages episodes, steps, rewards, and observations for all three DC cooling tasks.

Responsibilities:
  - Task registry (easy / medium / hard)
  - Episode lifecycle: reset(), step(), state()
  - Delegates physics to simulation.py
  - Delegates scoring to graders/
  - Constructs DCObservation from FacilityState
"""

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from .models import (
    DCAction,
    DCObservation,
    DCReward,
    ResetResult,
    StepResult,
    ZoneAdjustment,
    ZoneObservation,
)
from .simulation import (
    DCAction as SimDCAction,
    FacilityState,
    ZoneAdjustment as SimZoneAdjustment,
)
from .scenarios.easy import build_easy_scenario
from .scenarios.medium import build_medium_scenario
from .scenarios.hard import build_hard_scenario
from .graders.grader_easy import EasyGraderState
from .graders.grader_medium import MediumGrader
from .graders.grader_hard import HardGrader


# ── Constants ─────────────────────────────────────────────────────────────────

SAFE_TEMP_MIN = 18.0
SAFE_TEMP_MAX = 27.0

# Hard termination thresholds (per spec §4 Episode Boundaries)
MEDIUM_MAX_CONSECUTIVE_VIOLATIONS = 3       # any zone unsafe for 3+ consecutive steps
HARD_CRITICAL_TEMP_THRESHOLD = 32.0         # °C
HARD_CRITICAL_CONSECUTIVE_STEPS = 5         # sustained breach → episode ends, score = 0

# History buffer depth (for temporal observation)
HISTORY_BUFFER_DEPTH = 3

# Chiller fault observable: COP drop below this fraction of base triggers the flag
CHILLER_FAULT_COP_FRACTION = 0.60


def _reward_detail_as_dict(detail: Any) -> Dict[str, Any]:
    """Graders may return either a dict or a DCReward instance."""
    if isinstance(detail, DCReward):
        return detail.model_dump()
    if isinstance(detail, dict):
        return detail
    return {}


# ── Task registry ─────────────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy-single-zone": {
        "max_steps": 48,
        "scenario_builder": build_easy_scenario,
        "grader_class": EasyGraderState,
        "description": "Single-zone thermal runaway recovery under steady load",
        "hard_termination": False,
    },
    "medium-multi-zone": {
        "max_steps": 144,
        "scenario_builder": build_medium_scenario,
        "grader_class": MediumGrader,
        "description": "3-zone load surge with faulty sensor and diurnal variation",
        "hard_termination": True,
        "hard_term_mode": "violation_streak",   # 3+ consecutive steps any zone unsafe
    },
    "hard-cascading-failure": {
        "max_steps": 288,
        "scenario_builder": build_hard_scenario,
        "grader_class": HardGrader,
        "description": "4-zone cascading chiller failure with carbon-aware triage",
        "hard_termination": True,
        "hard_term_mode": "critical_breach",    # critical zone >32°C for 5+ steps → 0.0
    },
}


# ── Environment ───────────────────────────────────────────────────────────────

class DCEnvironment(Environment):
    """
    OpenEnv-compliant Data Centre environment — V2.

    Supports all three tasks via the TASK_CONFIGS registry.
    Physics delegation → simulation.py
    Scoring delegation → graders/
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy-single-zone", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Valid: {list(TASK_CONFIGS)}")
        self.task = task
        self.seed = seed
        self.config = TASK_CONFIGS[task]
        self.max_steps: int = self.config["max_steps"]

        # Runtime state (populated on reset)
        self._facility: Optional[FacilityState] = None
        self._grader: Optional[Any] = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_rewards: List[float] = []

        # Last action (needed for rate-limiting in simulation.step())
        self._last_action: Optional[SimDCAction] = None

        # Per-zone streak counters
        # consecutive_safe[zone_id]      = steps in safe band
        # consecutive_violation[zone_id] = steps outside safe band
        self._consecutive_safe: Dict[str, int] = {}
        self._consecutive_violation: Dict[str, int] = {}

        # Facility-wide SLA violation streak (any zone)
        self._sla_violation_streak: int = 0

        # Temporal history buffer: deque of per-step dicts (newest at right)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_BUFFER_DEPTH)

        # Base chiller COP (captured at reset for fault detection)
        self._base_chiller_cop: float = 3.5

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reset the environment and return initial observation."""
        seed = self.seed if self.seed is not None else random.randint(0, 99_999)
        self._facility = self.config["scenario_builder"](seed=seed)
        self._grader = self.config["grader_class"]()
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._history.clear()

        # Capture base COP for fault-detection heuristic
        self._base_chiller_cop = self._facility.chiller_cop

        # Initialise per-zone streak counters
        self._consecutive_safe = {z.zone_id: 0 for z in self._facility.zones}
        self._consecutive_violation = {z.zone_id: 0 for z in self._facility.zones}
        self._sla_violation_streak = 0

        # Seed a neutral last action so rate-limiting has a valid reference
        self._last_action = self._neutral_sim_action()

        obs = self._make_observation()
        return ResetResult(observation=obs, info={"task": self.task, "seed": seed})

    def step(self, action: DCAction) -> StepResult:
        """Apply agent action, advance simulation, compute reward, return StepResult."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._facility is None:
            raise RuntimeError("Call reset() before step().")

        # Convert Pydantic DCAction → simulation duck-type (rate-limiting compatible)
        sim_action = self._to_sim_action(action)

        # Advance simulation (rate-limiting + physics + time inside .step())
        step_info = self._facility.step(sim_action, self._last_action)
        # Apply diurnal weather curve if the scenario provided one
        if hasattr(self._facility, '_outside_temp_curve'):
            idx = min(self._step_count, len(self._facility._outside_temp_curve) - 1)
            self._facility.outside_temp_c = self._facility._outside_temp_curve[idx]
            self._facility.wet_bulb_temp_c = self._facility._wet_bulb_curve[idx]
        self._last_action = sim_action
        self._step_count += 1

        # Update per-zone streak counters
        any_violation = self._update_streaks()

        # Update facility-wide SLA violation streak
        if any_violation:
            self._sla_violation_streak += 1
        else:
            self._sla_violation_streak = 0

        # Push snapshot to history buffer
        self._push_history_snapshot()

        # Check hard termination conditions
        terminated_early, terminal_score = self._check_hard_termination()

        # Compute reward via grader
        grader_input = self._build_grader_input(action, step_info)
        step_reward, raw_reward_detail = self._grader.step(grader_input)
        reward_detail = _reward_detail_as_dict(raw_reward_detail)

        if terminated_early:
            step_reward = terminal_score
            self._done = True
        elif self._step_count >= self.max_steps:
            self._done = True

        self._episode_rewards.append(step_reward)

        bd = reward_detail.get("breakdown")
        if not isinstance(bd, dict):
            bd = reward_detail

        reward_model = DCReward(
            total=step_reward,
            temp_reward=reward_detail.get("temp_reward", 0.0),
            pue_reward=reward_detail.get("pue_reward", 0.0),
            carbon_reward=reward_detail.get("carbon_reward", 0.0),
            safety_penalty=reward_detail.get("safety_penalty", 0.0),
            roughness_penalty=reward_detail.get("roughness_penalty", 0.0),
            stability_bonus=reward_detail.get("stability_bonus", 0.0),
            # Legacy fields
            temperature_penalty=reward_detail.get("safety_penalty", 0.0),
            humidity_penalty=0.0,
            breakdown=bd,
        )

        info: Dict[str, Any] = {
            "action_clipped": step_info.get("action_clipped", {}),
            "terminated_early": terminated_early,
            "sla_violation_streak": self._sla_violation_streak,
            "consecutive_safe": dict(self._consecutive_safe),
            "consecutive_violation": dict(self._consecutive_violation),
        }
        if self._done:
            final_score = self._grader.final_score()
            info["final_score"] = final_score
            info["episode_rewards"] = list(self._episode_rewards)

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=step_reward,
            reward_detail=reward_model,
            done=self._done,
            info=info,
        )

    def state(self) -> dict:
        """Return full internal state for debugging/inspection."""
        if self._facility is None:
            return {"status": "not_initialized"}
        return {
            "task": self.task,
            "step": self._step_count,
            "done": self._done,
            "facility": self._facility.to_observation_dict(),
            "consecutive_safe": dict(self._consecutive_safe),
            "consecutive_violation": dict(self._consecutive_violation),
            "sla_violation_streak": self._sla_violation_streak,
            "episode_rewards": list(self._episode_rewards),
            "history_depth": len(self._history),
        }

    # ── Streak tracking ────────────────────────────────────────────────────────

    def _update_streaks(self) -> bool:
        """
        Update per-zone consecutive safe / violation counters.

        Returns True if any zone is currently in violation.
        """
        any_violation = False
        for zone in self._facility.zones:
            zid = zone.zone_id
            in_safe = SAFE_TEMP_MIN <= zone.temp_c <= SAFE_TEMP_MAX
            if in_safe:
                self._consecutive_safe[zid] = self._consecutive_safe.get(zid, 0) + 1
                self._consecutive_violation[zid] = 0
            else:
                self._consecutive_violation[zid] = self._consecutive_violation.get(zid, 0) + 1
                self._consecutive_safe[zid] = 0
                any_violation = True
        return any_violation

    # ── Hard termination ───────────────────────────────────────────────────────

    def _check_hard_termination(self):
        """
        Check task-specific hard termination conditions.

        Returns
        -------
        terminated : bool
        terminal_score : float  (0.0 on catastrophic failure, ignored if not terminated)
        """
        cfg = self.config
        if not cfg.get("hard_termination", False):
            return False, 0.0

        mode = cfg.get("hard_term_mode", "")

        if mode == "violation_streak":
            # Medium task: any zone unsafe for 3+ consecutive steps → terminate
            if self._sla_violation_streak >= MEDIUM_MAX_CONSECUTIVE_VIOLATIONS:
                return True, 0.0

        elif mode == "critical_breach":
            # Hard task: any critical zone > 32°C for 5+ consecutive steps → score = 0
            for zone in self._facility.zones:
                if zone.zone_priority < 2:   # only critical zones (priority == 2)
                    continue
                zid = zone.zone_id
                if (
                    zone.temp_c > HARD_CRITICAL_TEMP_THRESHOLD
                    and self._consecutive_violation.get(zid, 0) >= HARD_CRITICAL_CONSECUTIVE_STEPS
                ):
                    return True, 0.0

        return False, 0.0

    # ── History buffer ─────────────────────────────────────────────────────────

    def _push_history_snapshot(self):
        """Append a compact per-zone snapshot to the rolling history buffer."""
        snapshot = {
            "step": self._step_count,
            "pue": self._facility.pue,
            "zones": {
                z.zone_id: {
                    "cold_aisle_temp_c": z.temp_c,
                    "hot_aisle_temp_c": z.hot_aisle_temp_c,
                    "fan_speed_pct": z.fan_speed_pct,
                }
                for z in self._facility.zones
            },
        }
        self._history.append(snapshot)

    def _history_as_list(self) -> List[Dict[str, Any]]:
        """Return history buffer as a list ordered oldest → newest (t-3, t-2, t-1)."""
        return list(self._history)

    # ── Grader input construction ──────────────────────────────────────────────

    def _build_grader_input(self, action: DCAction, step_info: Dict) -> Dict[str, Any]:
        """
        Assemble all data the grader needs for one step.

        Graders are stateless between calls — they receive everything here.
        """
        f = self._facility
        return {
            "step": self._step_count,
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "temp_c": z.temp_c,
                    "hot_aisle_temp_c": z.hot_aisle_temp_c,
                    "it_load_kw": z.it_load_kw,
                    "fan_speed_pct": z.fan_speed_pct,
                    "supply_air_temp_setpoint_c": z.supply_air_temp_setpoint_c,
                    "zone_priority": z.zone_priority,
                    "consecutive_safe": self._consecutive_safe.get(z.zone_id, 0),
                    "consecutive_violation": self._consecutive_violation.get(z.zone_id, 0),
                }
                for z in f.zones
            ],
            "current_pue": f.pue,
            "pid_baseline_pue": f.pid_baseline_pue,
            "carbon_intensity_normalized": f.grid_carbon_intensity_normalized,
            "carbon_intensity_label": f.grid_carbon_intensity,
            "chiller_active": f.chiller_active,
            "chiller_setpoint_c": f.chiller_setpoint_c,
            "sla_violation_streak": self._sla_violation_streak,
            "action": action,
            "last_action": self._last_action,
            "action_clipped": step_info.get("action_clipped", {}),
            "reasoning": action.reasoning,
        }

    # ── Observation construction ───────────────────────────────────────────────

    def _make_observation(self) -> DCObservation:
        """Convert FacilityState → DCObservation (V2 full schema)."""
        f = self._facility
        raw = f.to_observation_dict()

        zone_obs = [
            ZoneObservation(
                zone_id=z["zone_id"],
                cold_aisle_temp_c=z["cold_aisle_temp_c"],
                hot_aisle_temp_c=z["hot_aisle_temp_c"],
                reported_temp_c=z["reported_temp_c"],
                supply_air_temp_c=z["supply_air_temp_c"],
                supply_air_temp_setpoint_c=z["supply_air_temp_setpoint_c"],
                it_load_kw=z["it_load_kw"],
                it_load_pct=z["it_load_pct"],
                fan_speed_pct=z["fan_speed_pct"],
                cooling_capacity_kw=z["cooling_capacity_kw"],
                humidity_pct=z["humidity_pct"],
                sensor_confidence=z["sensor_confidence"],
                zone_priority=z["zone_priority"],
                load_forecast_next_hour=self._forecast_load(z["zone_id"]),
            )
            for z in raw["zones"]
        ]

        return DCObservation(
            step=raw["step"],
            timestamp_hour=raw["timestamp_hour"],
            timestamp_day_sin=raw["timestamp_day_sin"],
            timestamp_day_cos=raw["timestamp_day_cos"],
            outside_temp_c=raw["outside_temp_c"],
            wet_bulb_temp_c=raw["wet_bulb_temp_c"],
            chiller_active=raw["chiller_active"],
            chiller_setpoint_c=raw["chiller_setpoint_c"],
            chiller_cop=raw["chiller_cop"],
            chiller_fault_detected=self._chiller_fault_detected(),
            ups_efficiency=raw["ups_efficiency"],
            current_pue=raw["current_pue"],
            free_cooling_potential=raw["free_cooling_potential"],
            grid_carbon_intensity=raw["grid_carbon_intensity"],
            carbon_intensity_normalized=raw["grid_carbon_intensity_normalized"],
            load_curve_phase=self._load_curve_phase(),
            zones=zone_obs,
            history=self._history_as_list(),
            sla_violation_streak=self._sla_violation_streak,
            maintenance_active=any(
                "maintenance" in note.lower()
                for note in raw.get("maintenance_notes", [])
            ),
            maintenance_notes=raw.get("maintenance_notes", []),
            upcoming_events=raw.get("upcoming_events", []),
        )

    # ── Helper: chiller fault detection (observable) ───────────────────────────

    def _chiller_fault_detected(self) -> bool:
        """
        Returns True if the chiller COP has dropped below 60 % of its baseline —
        an observable anomaly signal (not ground truth).
        """
        if not self._facility.chiller_active:
            return True
        if self._base_chiller_cop <= 0:
            return False
        current_cop = self._facility.chiller_cop
        return current_cop < (self._base_chiller_cop * CHILLER_FAULT_COP_FRACTION)

    # ── Helper: load curve phase ───────────────────────────────────────────────

    def _load_curve_phase(self) -> str:
        """
        Classify the current hour into a diurnal load phase.

        ramp_up   : 06:00–10:00
        peak      : 10:00–17:00
        ramp_down : 17:00–22:00
        idle      : 22:00–06:00
        """
        hour = self._facility.timestamp_hour
        if 6 <= hour < 10:
            return "ramp_up"
        elif 10 <= hour < 17:
            return "peak"
        elif 17 <= hour < 22:
            return "ramp_down"
        else:
            return "idle"

    # ── Helper: load forecast ──────────────────────────────────────────────────

    def _forecast_load(self, zone_id: str) -> float:
        """
        Simple 1-hour-ahead load forecast using the facility's load curve.

        Returns predicted IT load in kW for the given zone.
        """
        f = self._facility
        if not f.load_curve:
            return 0.0

        zone = next((z for z in f.zones if z.zone_id == zone_id), None)
        if zone is None:
            return 0.0

        future_hour = (f.timestamp_hour + 1.0) % 24.0
        future_idx = int(future_hour) % 24
        future_normalised = f.load_curve[future_idx]
        return round(zone.base_it_load_kw * future_normalised, 1)

    # ── Action conversion ──────────────────────────────────────────────────────

    def _to_sim_action(self, action: DCAction) -> SimDCAction:
        """Convert Pydantic DCAction → simulation duck-type SimDCAction."""
        sim_adjustments = [
            SimZoneAdjustment(
                zone_id=adj.zone_id,
                fan_speed_pct=adj.fan_speed_pct,
                supply_air_temp_setpoint_c=adj.supply_air_temp_setpoint_c,
            )
            for adj in action.zone_adjustments
        ]
        return SimDCAction(
            zone_adjustments=sim_adjustments,
            chiller_setpoint_c=action.chiller_setpoint_c,
            chiller_active=action.chiller_active,
            reasoning=action.reasoning,
        )

    def _neutral_sim_action(self) -> SimDCAction:
        """
        Build a neutral (do-nothing) SimDCAction from the current facility state.
        Used as the reference point for rate-limiting on the first real step.
        """
        if self._facility is None:
            return SimDCAction(zone_adjustments=[], chiller_setpoint_c=10.0, chiller_active=True)

        sim_adjustments = [
            SimZoneAdjustment(
                zone_id=z.zone_id,
                fan_speed_pct=z.fan_speed_pct,
                supply_air_temp_setpoint_c=z.supply_air_temp_setpoint_c,
            )
            for z in self._facility.zones
        ]
        return SimDCAction(
            zone_adjustments=sim_adjustments,
            chiller_setpoint_c=self._facility.chiller_setpoint_c,
            chiller_active=self._facility.chiller_active,
        )