"""
Grader for Task 1 (Easy): Single-zone temperature control.

Scoring criteria (deterministic, reproducible):
  - Temperature compliance: zone must stay in [18, 27] °C
  - PUE improvement: reward for reducing PUE from baseline (1.6)
  - Stability bonus: sustained compliance across multiple steps

Score is always in [0.0, 1.0].
"""

from typing import List
from dataclasses import dataclass

TEMP_MIN = 18.0
TEMP_MAX = 27.0
BASELINE_PUE = 1.65        # Starting PUE at fan_speed=60%
IDEAL_PUE = 1.15           # Best achievable PUE for this scenario
MAX_STEPS = 12


@dataclass
class EasyGraderState:
    steps_in_range: int = 0
    steps_total: int = 0
    pue_readings: List[float] = None
    temp_readings: List[float] = None

    def __post_init__(self):
        if self.pue_readings is None:
            self.pue_readings = []
        if self.temp_readings is None:
            self.temp_readings = []


def compute_step_reward(
    zone_temp: float,
    current_pue: float,
    grader_state: EasyGraderState,
) -> tuple:
    """
    Returns (reward: float, reward_detail: dict)
    reward is in [-1, +1] per step.
    """
    grader_state.steps_total += 1
    grader_state.pue_readings.append(current_pue)
    grader_state.temp_readings.append(zone_temp)

    # ── Temperature component ─────────────────────────────────────────────────
    in_range = TEMP_MIN <= zone_temp <= TEMP_MAX

    if in_range:
        grader_state.steps_in_range += 1
        # Bonus for being close to ideal (22°C)
        closeness = 1.0 - abs(zone_temp - 22.0) / 5.0
        temp_reward = 0.4 + 0.1 * closeness
    else:
        overshoot = max(0, zone_temp - TEMP_MAX)
        undershoot = max(0, TEMP_MIN - zone_temp)
        violation = overshoot + undershoot
        temp_reward = -0.3 * min(violation / 3.0, 1.0)  # penalty, capped

    # ── PUE component ─────────────────────────────────────────────────────────
    pue_normalized = (BASELINE_PUE - current_pue) / (BASELINE_PUE - IDEAL_PUE)
    pue_normalized = max(-0.5, min(1.0, pue_normalized))
    pue_reward = 0.4 * pue_normalized

    # ── Humidity penalty (minor) ───────────────────────────────────────────────
    hum_penalty = 0.0  # handled in observation; not passed here for easy task

    total = round(temp_reward + pue_reward, 4)
    total = max(-1.0, min(1.0, total))

    detail = {
        "temp_reward": round(temp_reward, 4),
        "pue_reward": round(pue_reward, 4),
        "in_range": in_range,
        "pue": round(current_pue, 4),
        "zone_temp": round(zone_temp, 2),
    }
    return total, detail


def compute_final_score(grader_state: EasyGraderState) -> float:
    """
    Final score in [0.0, 1.0] after episode ends.

    Breakdown:
      60% — fraction of steps where temperature was in safe range
      40% — average PUE improvement vs baseline
    """
    if grader_state.steps_total == 0:
        return 0.0

    # Temperature compliance fraction
    compliance = grader_state.steps_in_range / grader_state.steps_total

    # Average PUE score
    avg_pue = sum(grader_state.pue_readings) / len(grader_state.pue_readings)
    pue_score = (BASELINE_PUE - avg_pue) / (BASELINE_PUE - IDEAL_PUE)
    pue_score = max(0.0, min(1.0, pue_score))

    score = 0.60 * compliance + 0.40 * pue_score
    return round(score, 4)