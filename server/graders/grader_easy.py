"""
Grader for Task 1 (Easy): Single-zone thermal runaway recovery.

Interface contract (must match environment.py):
  - EasyGraderState() is instantiated by environment on reset()
  - grader.step(grader_input: dict) -> (float, DCReward)  called every step
  - grader.final_score() -> float                          called at episode end

grader_input keys (provided by environment._build_grader_input):
  step, zones, current_pue, pid_baseline_pue,
  carbon_intensity_normalized, carbon_intensity_label,
  chiller_active, chiller_setpoint_c, sla_violation_streak,
  action, last_action, action_clipped, reasoning

Scoring criteria (deterministic, reproducible, 0.0–1.0):
  60% — fraction of steps where cold_aisle_temp ∈ [18, 27] °C
  40% — average PUE improvement vs pid_baseline_pue (NOT vs zero-cooling)
"""

from typing import Any, Dict, List, Tuple

from ..models import DCReward


# ── Constants ──────────────────────────────────────────────────────────────────

TEMP_MIN   = 18.0
TEMP_MAX   = 27.0
TEMP_IDEAL = 22.0

# Best physically achievable PUE on the easy scenario.
# PUE reward is measured against pid_baseline_pue supplied per-step from
# the facility (pre-computed at scenario build time), not against this constant.
# This is kept only as a floor for normalisation.
IDEAL_PUE = 1.18


# ── Grader class ───────────────────────────────────────────────────────────────

class EasyGraderState:
    """
    Stateful grader for the easy-single-zone task.

    Instantiated fresh on every reset(). Accumulates per-step metrics
    and produces a final score in [0.0, 1.0] at episode end.
    """

    def __init__(self):
        self.steps_in_range:   int         = 0
        self.steps_total:      int         = 0
        self.pue_readings:     List[float] = []
        self.temp_readings:    List[float] = []
        self.consecutive_safe: int         = 0

    # ── Step interface (called by environment.step()) ──────────────────────────

    def step(self, grader_input: Dict[str, Any]) -> Tuple[float, DCReward]:
        """
        Compute per-step reward from grader_input dict.

        Returns
        -------
        (total_reward: float, reward_detail: DCReward)
        """
        self.steps_total += 1
        current_pue      = grader_input["current_pue"]
        pid_baseline_pue = grader_input.get("pid_baseline_pue", 1.55)
        self._pid_baseline_pue = pid_baseline_pue
        # Easy task: always single zone — take first zone
        zones = grader_input["zones"]
        zone  = zones[0] if zones else {}

        zone_temp        = zone.get("temp_c", 30.0)
        consecutive_safe = zone.get("consecutive_safe", self.consecutive_safe)

        self.pue_readings.append(current_pue)
        self.temp_readings.append(zone_temp)

        # ── Temperature reward ─────────────────────────────────────────────────
        in_range = TEMP_MIN <= zone_temp <= TEMP_MAX

        if in_range:
            self.steps_in_range  += 1
            self.consecutive_safe = consecutive_safe
            closeness    = 1.0 - abs(zone_temp - TEMP_IDEAL) / 5.0
            temp_reward  = 0.40 + 0.10 * closeness
            # Compounding stability bonus for sustained streaks
            streak_bonus = 0.05 * min(self.consecutive_safe / 10.0, 1.0)
            temp_reward  = min(0.55, temp_reward + streak_bonus)
        else:
            self.consecutive_safe = 0
            overshoot   = max(0.0, zone_temp - TEMP_MAX)
            undershoot  = max(0.0, TEMP_MIN - zone_temp)
            violation   = overshoot + undershoot
            temp_reward = -0.30 * min(violation / 3.0, 1.0)

        # ── PUE reward (vs PID baseline, not vs zero-cooling) ─────────────────
        # Suppressed while zone is out of range: don't penalise necessary aggressive cooling.
        pue_range  = max(pid_baseline_pue - IDEAL_PUE, 0.01)   # avoid div/0
        pue_vs_pid = (pid_baseline_pue - current_pue) / pue_range
        pue_vs_pid = max(-1.0, min(1.0, pue_vs_pid))
        pue_reward = 0.35 * pue_vs_pid if in_range else 0.0

        # ── Carbon signal (light penalty, easy task doesn't heavily weight it) ─
        carbon = grader_input.get("carbon_intensity_normalized", 0.5)
        # Approximate cooling power from PUE and IT load
        total_it_kw = sum(z.get("it_load_kw", 0.0) for z in zones)
        cooling_power_est = max(0.0, (current_pue - 1.0) * total_it_kw)
        carbon_reward = -0.05 * (cooling_power_est / max(total_it_kw, 1.0)) * carbon

        # ── Total ──────────────────────────────────────────────────────────────
        total = round(
            max(-1.0, min(1.0, temp_reward + pue_reward + carbon_reward)), 4
        )

        reward_detail = DCReward(
            total=total,
            temp_reward=round(temp_reward, 4),
            pue_reward=round(pue_reward, 4),
            carbon_reward=round(carbon_reward, 4),
            safety_penalty=0.0,
            roughness_penalty=0.0,
            stability_bonus=round(
                0.05 * min(self.consecutive_safe / 10.0, 1.0) if in_range else 0.0, 4
            ),
            temperature_penalty=round(min(0.0, temp_reward), 4),
            humidity_penalty=0.0,
            breakdown={
                "temp_reward":   round(temp_reward, 4),
                "pue_reward":    round(pue_reward, 4),
                "carbon_reward": round(carbon_reward, 4),
                "in_range":      float(in_range),
                "pue":           round(current_pue, 4),
                "pue_vs_pid":    round(pue_vs_pid, 4),
                "zone_temp":     round(zone_temp, 2),
            },
        )

        return total, reward_detail

    # ── Final score (called by inference.py after episode ends) ───────────────

    def final_score(self) -> float:
        """
        Final episode score in [0.0, 1.0].

        60% — temperature compliance fraction
        40% — average PUE improvement vs pid_baseline_pue
        """
        if self.steps_total == 0:
            return 0.0

        compliance = self.steps_in_range / self.steps_total

        avg_pue      = sum(self.pue_readings) / len(self.pue_readings)
        # Use the first pid_baseline_pue seen; fall back to 1.55 if not recorded
        pid_ref      = getattr(self, "_pid_baseline_pue", 1.55)
        pue_range    = max(pid_ref - IDEAL_PUE, 0.01)
        pue_score    = (pid_ref - avg_pue) / pue_range
        pue_score    = max(0.0, min(1.0, pue_score))

        return round(0.60 * compliance + 0.40 * pue_score, 4)