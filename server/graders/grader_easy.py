"""
Grader for Task 1 (Easy): Single-zone thermal runaway recovery.

Scoring criteria (deterministic, reproducible):
  - Temperature compliance: zone must stay in [18, 27] C
  - PUE improvement: reward relative to PID baseline (NOT vs zero-cooling)
  - Stability bonus: sustained compliance across multiple steps

FIX: PUE score is now benchmarked against PID_BASELINE_PUE (the PUE a simple
proportional controller achieves on this scenario). Previously it was measured
against zero-cooling which gave a perfect PUE score to agents that did nothing.

Score is always in [0.0, 1.0].
"""

from typing import List
from dataclasses import dataclass, field

TEMP_MIN     = 18.0
TEMP_MAX     = 27.0
TEMP_IDEAL   = 22.0

# PUE a simple proportional (PID) controller achieves on the easy scenario.
# Pre-computed by running pid_baseline.py on build_easy_scenario(seed=0..99).
# Average PID PUE on this scenario ≈ 1.55 (fan ~70%, chiller active at 10C setpoint).
# An agent must beat THIS to earn PUE reward, not beat zero-cooling.
PID_BASELINE_PUE = 1.55

# Best physically achievable PUE on this scenario (fan ~40%, chiller optimised).
IDEAL_PUE = 1.18

MAX_STEPS = 48


@dataclass
class EasyGraderState:
    steps_in_range:  int         = 0
    steps_total:     int         = 0
    pue_readings:    List[float] = field(default_factory=list)
    temp_readings:   List[float] = field(default_factory=list)
    consecutive_safe: int        = 0   # current streak of in-range steps


def compute_step_reward(
    zone_temp:     float,
    current_pue:   float,
    grader_state:  EasyGraderState,
) -> tuple:
    """
    Returns (reward: float, detail: dict).
    Reward is in [-1.0, +1.0] per step.
    """
    grader_state.steps_total    += 1
    grader_state.pue_readings.append(current_pue)
    grader_state.temp_readings.append(zone_temp)

    # ── Temperature component ─────────────────────────────────────────────────
    in_range = TEMP_MIN <= zone_temp <= TEMP_MAX

    if in_range:
        grader_state.steps_in_range  += 1
        grader_state.consecutive_safe += 1
        closeness   = 1.0 - abs(zone_temp - TEMP_IDEAL) / 5.0
        temp_reward = 0.40 + 0.10 * closeness
        # Small stability bonus for sustained streaks
        streak_bonus = 0.05 * min(grader_state.consecutive_safe / 10.0, 1.0)
        temp_reward  = min(0.55, temp_reward + streak_bonus)
    else:
        grader_state.consecutive_safe = 0
        overshoot  = max(0.0, zone_temp - TEMP_MAX)
        undershoot = max(0.0, TEMP_MIN - zone_temp)
        violation  = overshoot + undershoot
        temp_reward = -0.30 * min(violation / 3.0, 1.0)

    # ── PUE component (FIX: benchmarked against PID baseline) ─────────────────
    # positive when agent beats PID; negative when agent is worse than PID
    pue_range     = PID_BASELINE_PUE - IDEAL_PUE          # e.g. 1.55 - 1.18 = 0.37
    pue_vs_pid    = (PID_BASELINE_PUE - current_pue) / pue_range
    pue_vs_pid    = max(-1.0, min(1.0, pue_vs_pid))
    pue_reward    = 0.35 * pue_vs_pid

    total = round(temp_reward + pue_reward, 4)
    total = max(-1.0, min(1.0, total))

    detail = {
        "temp_reward":  round(temp_reward, 4),
        "pue_reward":   round(pue_reward,  4),
        "in_range":     in_range,
        "pue":          round(current_pue, 4),
        "zone_temp":    round(zone_temp,   2),
        "pue_vs_pid":   round(pue_vs_pid,  4),
    }
    return total, detail


def compute_final_score(grader_state: EasyGraderState) -> float:
    """
    Final episode score in [0.0, 1.0].

    Breakdown:
      60% — fraction of steps where temperature was in safe range [18, 27] C
      40% — average PUE improvement vs PID baseline (not vs zero-cooling)
    """
    if grader_state.steps_total == 0:
        return 0.0

    # Temperature compliance fraction
    compliance = grader_state.steps_in_range / grader_state.steps_total

    # Average PUE vs PID baseline
    avg_pue   = sum(grader_state.pue_readings) / len(grader_state.pue_readings)
    pue_range = PID_BASELINE_PUE - IDEAL_PUE
    pue_score = (PID_BASELINE_PUE - avg_pue) / pue_range
    pue_score = max(0.0, min(1.0, pue_score))

    score = 0.60 * compliance + 0.40 * pue_score
    return round(score, 4)