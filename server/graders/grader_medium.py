"""
Grader for Task 2 (Medium): Multi-zone load surge with faulty sensor.

Scoring criteria (deterministic, reproducible):
  - All-zone temperature compliance: every zone in [18, 27] °C
  - PUE improvement vs PID baseline
  - Sensor inference quality: did the agent act correctly for zone_ai despite bad sensor?
  - Peak-load compliance: were all zones safe during the high-load surge window?

Final score breakdown [0.0-1.0]:
  35% — All-zone temperature compliance fraction
  25% — Average PUE improvement vs pid_baseline_pue
  20% — Sensor inference quality for zone_ai (faulty sensor zone)
  20% — Compliance fraction during peak load window (steps 30-80)

Sensor inference quality:
  A naive agent trusts the faulty sensor (reports +9-12°C above actual).
  It will over-cool zone_ai, wasting energy, or panic and max fans unnecessarily.
  We measure quality by comparing the agent's supply_air_temp_setpoint_c for zone_ai
  against what an oracle (knowing true temp) would set. Closer = better inference.
  We also reward the agent for NOT over-cooling when sensor_confidence is low.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Temperature bounds ────────────────────────────────────────────────────────
TEMP_MIN   = 18.0
TEMP_MAX   = 27.0
TEMP_IDEAL = 22.0

# ── PUE ───────────────────────────────────────────────────────────────────────
IDEAL_PUE = 1.15

# ── Peak load window (steps where IT load surges) ─────────────────────────────
PEAK_LOAD_START = 30
PEAK_LOAD_END   = 80

# ── Faulty sensor zone ────────────────────────────────────────────────────────
FAULTY_ZONE_ID = "zone_ai"

# ── Oracle supply setpoint: what a perfect agent would target ─────────────────
# When true temp is in range and load is high, oracle targets ~20°C supply air.
ORACLE_SUPPLY_SETPOINT_HIGH_LOAD = 20.0
ORACLE_SUPPLY_SETPOINT_NORMAL    = 22.0

# ── Priority multipliers for step reward ─────────────────────────────────────
PRIORITY_TEMP_MULTIPLIER = {0: 0.7, 1: 1.0, 2: 1.4}
PRIORITY_VIOLATION_MULTIPLIER = {0: 1.0, 1: 1.5, 2: 2.5}

# ── Final score weights ───────────────────────────────────────────────────────
FINAL_COMPLIANCE_WEIGHT     = 0.35
FINAL_PUE_WEIGHT            = 0.25
FINAL_SENSOR_QUALITY_WEIGHT = 0.20
FINAL_PEAK_WEIGHT           = 0.20

# ── Step reward weights ───────────────────────────────────────────────────────
STEP_TEMP_WEIGHT   = 0.50
STEP_PUE_WEIGHT    = 0.25
STEP_CARBON_WEIGHT = 0.15
STEP_ROUGH_WEIGHT  = 0.10


# ── Grader ────────────────────────────────────────────────────────────────────

@dataclass
class MediumGrader:
    """
    Stateful grader for the medium task.

    Interface contract:
      grader.step(grader_input: dict) -> (reward: float, detail: dict)
      grader.final_score()            -> float
    """

    # ── Aggregate counters ────────────────────────────────────────────────────
    steps_all_zones_safe: int   = 0
    steps_total:          int   = 0

    # ── PUE tracking ─────────────────────────────────────────────────────────
    pue_readings: List[float] = field(default_factory=list)

    # ── Sensor inference quality ──────────────────────────────────────────────
    # Per step: |agent_supply_setpoint - oracle_supply_setpoint| for zone_ai.
    # Lower error = agent is implicitly inferring true temp correctly.
    zone_ai_supply_errors: List[float] = field(default_factory=list)

    # ── Peak load window tracking ─────────────────────────────────────────────
    peak_load_steps_total:  int = 0
    peak_load_steps_safe:   int = 0

    # ── Carbon-weighted cooling (for reference / debugging) ───────────────────
    carbon_weighted_cooling: List[float] = field(default_factory=list)

    # ── Internal ──────────────────────────────────────────────────────────────
    _pid_baseline_pue: float = 1.55

    def step(self, grader_input: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute step reward from grader_input dict.

        Expected keys (set by environment._build_grader_input):
          step                   : int
          zones                  : List[dict] — each has zone_id, temp_c, zone_priority,
                                   fan_speed_pct, supply_air_temp_setpoint_c,
                                   consecutive_safe, consecutive_violation, it_load_kw
          current_pue            : float
          pid_baseline_pue       : float
          carbon_intensity_normalized : float
          action                 : DCAction  (Pydantic model)
          last_action            : SimDCAction duck-type
          action_clipped         : dict
        """
        # ── Unpack facility-level inputs ──────────────────────────────────────
        step_num: int        = grader_input.get("step", 0)
        current_pue: float   = grader_input["current_pue"]
        pid_baseline_pue: float = grader_input.get("pid_baseline_pue", self._pid_baseline_pue)
        self._pid_baseline_pue  = pid_baseline_pue
        carbon: float        = grader_input.get("carbon_intensity_normalized", 0.5)
        zones: List[dict]    = grader_input.get("zones", [])
        action               = grader_input.get("action")
        last_action          = grader_input.get("last_action")

        # ── Accumulate ────────────────────────────────────────────────────────
        self.steps_total += 1
        self.pue_readings.append(current_pue)

        in_peak = PEAK_LOAD_START <= step_num <= PEAK_LOAD_END

        # ── Per-zone temperature reward ───────────────────────────────────────
        temp_reward_total = 0.0
        all_zones_safe    = True

        for zone in zones:
            zone_temp: float = zone["temp_c"]
            priority: int    = zone.get("zone_priority", 1)
            consec_safe: int = zone.get("consecutive_safe", 0)
            in_range = TEMP_MIN <= zone_temp <= TEMP_MAX

            if in_range:
                closeness = 1.0 - abs(zone_temp - TEMP_IDEAL) / 5.0
                base = 0.5 + 0.3 * closeness
                temp_reward_total += base * PRIORITY_TEMP_MULTIPLIER[priority]
            else:
                all_zones_safe = False
                violation = max(zone_temp - TEMP_MAX, TEMP_MIN - zone_temp)
                base = -0.4 * min(violation / 5.0, 1.0)
                temp_reward_total += base * PRIORITY_VIOLATION_MULTIPLIER[priority]

        # Normalise across zones
        n_zones = max(len(zones), 1)
        temp_reward = STEP_TEMP_WEIGHT * (temp_reward_total / n_zones)

        # Compliance counters
        if all_zones_safe:
            self.steps_all_zones_safe += 1
        if in_peak:
            self.peak_load_steps_total += 1
            if all_zones_safe:
                self.peak_load_steps_safe += 1

        # ── PUE component ─────────────────────────────────────────────────────
        # Suppressed while any zone is out of range: don't penalise recovery cooling.
        denominator = max(pid_baseline_pue - IDEAL_PUE, 0.01)
        pue_improvement = (pid_baseline_pue - current_pue) / denominator
        pue_improvement = max(-0.5, min(1.0, pue_improvement))
        pue_reward = STEP_PUE_WEIGHT * pue_improvement if all_zones_safe else 0.0

        # ── Carbon component ──────────────────────────────────────────────────
        # Total cooling power proxied by sum of fan speed × capacity fractions
        total_cooling_proxy = sum(
            z.get("fan_speed_pct", 50.0) / 100.0 for z in zones
        ) / n_zones
        carbon_cost = total_cooling_proxy * carbon
        self.carbon_weighted_cooling.append(carbon_cost)
        carbon_reward = -STEP_CARBON_WEIGHT * carbon_cost

        # ── Action roughness ──────────────────────────────────────────────────
        roughness_penalty = _compute_roughness(action, last_action, zones)
        roughness_reward  = -STEP_ROUGH_WEIGHT * roughness_penalty

        # ── Sensor inference quality for zone_ai ──────────────────────────────
        supply_error = _compute_sensor_inference_error(action, zones, step_num)
        if supply_error is not None:
            self.zone_ai_supply_errors.append(supply_error)

        # ── Stability bonus ───────────────────────────────────────────────────
        # Average consecutive safe steps across all zones
        avg_consec_safe = (
            sum(z.get("consecutive_safe", 0) for z in zones) / n_zones
        )
        stability_bonus = 0.05 * min(avg_consec_safe / 10.0, 1.0)

        # ── Combine ───────────────────────────────────────────────────────────
        total = round(
            temp_reward + pue_reward + carbon_reward + roughness_reward + stability_bonus,
            4,
        )
        total = max(-1.0, min(1.0, total))

        detail: Dict[str, Any] = {
            "temp_reward":       round(temp_reward, 4),
            "pue_reward":        round(pue_reward, 4),
            "carbon_reward":     round(carbon_reward, 4),
            "roughness_penalty": round(roughness_reward, 4),
            "stability_bonus":   round(stability_bonus, 4),
            "safety_penalty":    0.0,   # folded into temp_reward for medium task
            "all_zones_safe":    all_zones_safe,
            "in_peak_window":    in_peak,
            "pue":               round(current_pue, 4),
            "pid_baseline_pue":  round(pid_baseline_pue, 4),
            "pue_improvement":   round(pue_improvement, 4),
            "supply_error_zone_ai": round(supply_error, 4) if supply_error is not None else None,
        }
        return total, detail

    def final_score(self) -> float:
        """
        Final episode score in [0.0, 1.0].

        Breakdown:
          35% — all-zone temperature compliance fraction
          25% — average PUE improvement vs pid_baseline_pue
          20% — sensor inference quality (zone_ai supply setpoint accuracy)
          20% — compliance fraction during peak load window
        """
        if self.steps_total == 0:
            return 0.0

        # ── Temperature compliance ────────────────────────────────────────────
        compliance = self.steps_all_zones_safe / self.steps_total

        # ── PUE improvement ───────────────────────────────────────────────────
        avg_pue = sum(self.pue_readings) / len(self.pue_readings)
        denominator = max(self._pid_baseline_pue - IDEAL_PUE, 0.01)
        pue_score = (self._pid_baseline_pue - avg_pue) / denominator
        pue_score = max(0.0, min(1.0, pue_score))

        # ── Sensor inference quality ──────────────────────────────────────────
        # Max tolerable supply setpoint error: 6°C (range is 16–26, so 6°C is large)
        if self.zone_ai_supply_errors:
            avg_error = sum(self.zone_ai_supply_errors) / len(self.zone_ai_supply_errors)
            sensor_score = max(0.0, 1.0 - avg_error / 6.0)
        else:
            sensor_score = 0.5   # no data → neutral score

        # ── Peak load compliance ──────────────────────────────────────────────
        if self.peak_load_steps_total > 0:
            peak_score = self.peak_load_steps_safe / self.peak_load_steps_total
        else:
            peak_score = compliance   # no peak window observed → use overall compliance

        score = (
            FINAL_COMPLIANCE_WEIGHT     * compliance
            + FINAL_PUE_WEIGHT          * pue_score
            + FINAL_SENSOR_QUALITY_WEIGHT * sensor_score
            + FINAL_PEAK_WEIGHT         * peak_score
        )
        return round(score, 4)


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_roughness(action: Any, last_action: Any, zones: List[dict]) -> float:
    """
    Measure action roughness as a normalised [0-1] value.

    Returns 0.0 if last_action is unavailable (first step).
    """
    if action is None or last_action is None:
        return 0.0

    # Chiller setpoint delta
    try:
        chiller_delta = abs(action.chiller_setpoint_c - last_action.chiller_setpoint_c)
    except AttributeError:
        chiller_delta = 0.0

    # Per-zone fan and supply temp deltas
    last_zone_map = {
        adj.zone_id: adj
        for adj in getattr(last_action, "zone_adjustments", [])
    }
    fan_deltas     = []
    supply_deltas  = []

    for adj in getattr(action, "zone_adjustments", []):
        last_adj = last_zone_map.get(adj.zone_id)
        if last_adj is None:
            continue
        fan_deltas.append(abs(adj.fan_speed_pct - last_adj.fan_speed_pct))
        supply_deltas.append(
            abs(adj.supply_air_temp_setpoint_c - last_adj.supply_air_temp_setpoint_c)
        )

    avg_fan_delta    = sum(fan_deltas) / max(len(fan_deltas), 1)
    avg_supply_delta = sum(supply_deltas) / max(len(supply_deltas), 1)

    roughness = (
        (avg_fan_delta / 100.0)  * 0.50
        + (avg_supply_delta / 10.0) * 0.30
        + (chiller_delta / 9.0)     * 0.20
    )
    return min(roughness, 1.0)


def _oracle_supply_setpoint(zone_temp: float, it_load_pct: float) -> float:
    """
    What a perfect agent knowing the TRUE zone temperature would set as
    supply_air_temp_setpoint_c for zone_ai.

    Oracle strategy:
      - If zone is warm (>23°C) or load is high (>0.85): target 20°C supply
      - Otherwise: target 22°C (efficient default)
    """
    if zone_temp > 23.0 or it_load_pct > 0.85:
        return ORACLE_SUPPLY_SETPOINT_HIGH_LOAD
    return ORACLE_SUPPLY_SETPOINT_NORMAL


def _compute_sensor_inference_error(
    action: Any,
    zones: List[dict],
    step_num: int,
) -> Optional[float]:
    """
    Compute the agent's supply setpoint error for zone_ai relative to oracle.

    Returns None if zone_ai is not in this episode's zone list or action is absent.

    The oracle knows the true temperature; we compare the agent's chosen
    supply_air_temp_setpoint_c against what the oracle would pick.
    A small error means the agent is implicitly reasoning about the true state
    even though the sensor is lying.
    """
    if action is None:
        return None

    # Find zone_ai in grader zone list
    ai_zone = next((z for z in zones if z.get("zone_id") == FAULTY_ZONE_ID), None)
    if ai_zone is None:
        return None

    true_temp: float   = ai_zone["temp_c"]
    it_load_pct: float = ai_zone.get("it_load_pct", 0.5)
    oracle_setpoint    = _oracle_supply_setpoint(true_temp, it_load_pct)

    # Find what the agent actually set for zone_ai
    agent_setpoint: Optional[float] = None
    for adj in getattr(action, "zone_adjustments", []):
        if getattr(adj, "zone_id", None) == FAULTY_ZONE_ID:
            agent_setpoint = getattr(adj, "supply_air_temp_setpoint_c", None)
            break

    if agent_setpoint is None:
        return None

    return abs(agent_setpoint - oracle_setpoint)