"""
Grader for Task 1 (Easy): Single-zone thermal runaway recovery.

Scoring criteria (deterministic, reproducible):
  - Temperature compliance: zone must stay in [18, 27] °C
  - PUE improvement: reward relative to PID baseline PUE (not a hardcoded constant)

Final score breakdown [0.0–1.0]:
  60% — fraction of steps where cold_aisle_temp ∈ [18, 27]
  40% — average PUE improvement vs pid_baseline_pue

Key V2 fix: pid_baseline_pue is pulled from the grader_input dict (sourced from
FacilityState.pid_baseline_pue) rather than a hardcoded BASELINE_PUE constant.
This makes the score meaningful relative to what a PID controller would achieve
on the same scenario, not relative to zero-cooling.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Temperature bounds ────────────────────────────────────────────────────────
TEMP_MIN = 18.0
TEMP_MAX = 27.0
TEMP_IDEAL = 22.0

# ── PUE bounds ────────────────────────────────────────────────────────────────
# IDEAL_PUE: best physically achievable PUE for this scenario.
# Used only to normalise the upper end of the PUE improvement scale.
# The lower reference (baseline) comes from FacilityState.pid_baseline_pue at runtime.
IDEAL_PUE = 1.15

# ── Reward weights (step reward) ──────────────────────────────────────────────
TEMP_WEIGHT = 0.60
PUE_WEIGHT  = 0.40

# ── Final score weights ───────────────────────────────────────────────────────
FINAL_COMPLIANCE_WEIGHT = 0.60
FINAL_PUE_WEIGHT        = 0.40


# ── Grader state ──────────────────────────────────────────────────────────────

@dataclass
class EasyGrader:
    """
    Stateful grader for the easy task.

    Accumulates per-step observations; computes final score on demand.
    Interface contract (matches environment.py expectations):
      grader.step(grader_input: dict) -> (reward: float, detail: dict)
      grader.final_score()            -> float
    """
    steps_in_range: int = 0
    steps_total:    int = 0
    pue_readings:   List[float] = field(default_factory=list)
    temp_readings:  List[float] = field(default_factory=list)

    # Captured from first step and held constant for the episode.
    # Overwritten each step in case the scenario updates it (it won't for easy,
    # but this keeps the grader defensive).
    _pid_baseline_pue: float = 1.55

    def step(self, grader_input: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute step reward from grader_input dict.

        Expected keys (set by environment._build_grader_input):
          zones            : list of zone dicts, each with temp_c, zone_priority, etc.
          current_pue      : float
          pid_baseline_pue : float  ← key V2 field
          (all others are ignored for the easy task)
        """
        # ── Unpack inputs ─────────────────────────────────────────────────────
        pid_baseline_pue: float = grader_input.get("pid_baseline_pue", self._pid_baseline_pue)
        self._pid_baseline_pue = pid_baseline_pue   # update in case scenario changes it

        current_pue: float = grader_input["current_pue"]

        # Easy task: single zone — take first zone
        zones = grader_input.get("zones", [])
        if not zones:
            return 0.0, {"error": "no zones in grader_input"}
        zone = zones[0]
        zone_temp: float = zone["temp_c"]

        # ── Accumulate ────────────────────────────────────────────────────────
        self.steps_total += 1
        self.pue_readings.append(current_pue)
        self.temp_readings.append(zone_temp)

        # ── Temperature component ─────────────────────────────────────────────
        in_range = TEMP_MIN <= zone_temp <= TEMP_MAX

        if in_range:
            self.steps_in_range += 1
            closeness = 1.0 - abs(zone_temp - TEMP_IDEAL) / 5.0
            temp_reward = TEMP_WEIGHT * (0.8 + 0.2 * closeness)   # [0.48–0.60]
        else:
            violation = max(zone_temp - TEMP_MAX, TEMP_MIN - zone_temp)
            temp_reward = -TEMP_WEIGHT * min(violation / 3.0, 1.0)  # [−0.60–0.00]

        # ── PUE component (benchmarked against PID baseline) ──────────────────
        denominator = max(pid_baseline_pue - IDEAL_PUE, 0.01)
        pue_improvement = (pid_baseline_pue - current_pue) / denominator
        pue_improvement = max(-0.5, min(1.0, pue_improvement))
        pue_reward = PUE_WEIGHT * pue_improvement                  # [−0.20–0.40]

        # ── Combine ───────────────────────────────────────────────────────────
        total = round(temp_reward + pue_reward, 4)
        total = max(-1.0, min(1.0, total))

        detail: Dict[str, Any] = {
            "temp_reward":       round(temp_reward, 4),
            "pue_reward":        round(pue_reward, 4),
            "safety_penalty":    0.0,
            "roughness_penalty": 0.0,
            "stability_bonus":   0.0,
            "carbon_reward":     0.0,
            "in_range":          in_range,
            "zone_temp":         round(zone_temp, 2),
            "pue":               round(current_pue, 4),
            "pid_baseline_pue":  round(pid_baseline_pue, 4),
            "pue_improvement":   round(pue_improvement, 4),
        }
        return total, detail

    def final_score(self) -> float:
        """
        Final episode score in [0.0, 1.0].

        Breakdown:
          60% — fraction of steps where cold_aisle_temp ∈ [18, 27]
          40% — average PUE improvement vs pid_baseline_pue
        """
        if self.steps_total == 0:
            return 0.0

        # Temperature compliance fraction
        compliance = self.steps_in_range / self.steps_total

        # Average PUE improvement vs PID baseline
        avg_pue = sum(self.pue_readings) / len(self.pue_readings)
        denominator = max(self._pid_baseline_pue - IDEAL_PUE, 0.01)
        pue_score = (self._pid_baseline_pue - avg_pue) / denominator
        pue_score = max(0.0, min(1.0, pue_score))

        score = FINAL_COMPLIANCE_WEIGHT * compliance + FINAL_PUE_WEIGHT * pue_score
        return round(score, 4)