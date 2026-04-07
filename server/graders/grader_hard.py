"""
Grader for Task 3 (Hard): Cascading chiller failure with carbon-aware triage.

Scoring criteria (deterministic, reproducible):
  - SLA compliance: critical zones (priority 2) never exceed 30°C for 5+ consecutive steps
  - Carbon efficiency: minimize cooling power during high-carbon grid windows
  - Recovery speed: how fast the agent restabilizes temperatures after chiller failure
  - Triage quality: did the agent sacrifice low-priority zones to protect critical ones?
  - Reasoning coherence: does the agent's stated reasoning match its actual actions?

Final score breakdown [0.0–1.0]:
  30% — SLA compliance (critical zone safety)
  25% — Carbon efficiency during high-carbon windows
  20% — Recovery speed after chiller failure (steps 8–16)
  15% — Triage quality
  10% — Reasoning coherence

Chiller failure timeline (condensed 40-step / 24-hr episode, step_scale=7.2):
  Step 3: COP begins degrading (FAULT_START_STEP; observable via chiller_fault_detected)
  Step 8: Chiller fully offline (CHILLER_OFFLINE_STEP = FAULT_START_STEP + 5 degradation steps)
  Steps 8–16: Recovery window (RECOVERY_WINDOW_END=16) — agent must restabilize using fans + free cooling only
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Temperature bounds ────────────────────────────────────────────────────────
TEMP_MIN           = 18.0
TEMP_MAX           = 27.0
TEMP_IDEAL         = 22.0
CRITICAL_THRESHOLD = 30.0   # hard SLA limit for critical zones
EMERGENCY_THRESHOLD = 35.0  # maximum penalty threshold

# ── Fault timeline ─────────────────────────────────────────────────────────────
FAULT_START_STEP    = 3     # COP begins degrading (was 15; rescaled for 40-step / 24-hr budget: 15/7.2 ≈ 2 → use 3 for pre-fault window)
CHILLER_OFFLINE_STEP = 8    # chiller fully offline (was 20; = FAULT_START_STEP + 5 degradation steps)
RECOVERY_WINDOW_END  = 16   # end of recovery assessment window (was 50; ≈ 40% of 40-step episode)

# ── Carbon thresholds ──────────────────────────────────────────────────────────
HIGH_CARBON_THRESHOLD = 0.55   # carbon_intensity_normalized above this = high carbon

# ── PUE ───────────────────────────────────────────────────────────────────────
IDEAL_PUE = 1.15

# ── Triage: which zone IDs are critical vs sacrificeable ──────────────────────
CRITICAL_ZONE_IDS    = {"zone_ai_1", "zone_ai_2"}
SACRIFICE_ZONE_IDS   = {"zone_infra"}   # zone_storage is PRIORITY_MEDIUM — not sacrificeable

# ── Priority safety: consecutive violation threshold ──────────────────────────
CRITICAL_CONSEC_VIOLATION_LIMIT = 5

# ── Final score weights ───────────────────────────────────────────────────────
FINAL_SLA_WEIGHT       = 0.30
FINAL_CARBON_WEIGHT    = 0.25
FINAL_RECOVERY_WEIGHT  = 0.20
FINAL_TRIAGE_WEIGHT    = 0.15
FINAL_REASONING_WEIGHT = 0.10

# ── Step reward weights ───────────────────────────────────────────────────────
STEP_TEMP_WEIGHT      = 0.45   # raised from 0.35 — temperature compliance is primary
STEP_PUE_WEIGHT       = 0.20
STEP_CARBON_WEIGHT    = 0.05   # lowered from 0.15 — don't override safety with carbon cost
STEP_SAFETY_WEIGHT    = 0.20
STEP_ROUGH_WEIGHT     = 0.05
STEP_STABILITY_WEIGHT = 0.05


# ── Grader ────────────────────────────────────────────────────────────────────

@dataclass
class HardGrader:
    """
    Stateful grader for the hard task.

    Interface contract:
      grader.step(grader_input: dict) -> (reward: float, detail: dict)
      grader.final_score()            -> float
    """

    # ── SLA / critical zone tracking ──────────────────────────────────────────
    critical_zone_violations: int  = 0    # total critical-zone steps above 30°C
    sla_terminated_early: bool     = False
    steps_total: int               = 0
    steps_critical_safe: int       = 0   # steps where ALL critical zones are safe

    # ── PUE by phase ──────────────────────────────────────────────────────────
    pre_fault_pue:  List[float] = field(default_factory=list)   # steps 0–2  (before FAULT_START_STEP=3)
    post_fault_pue: List[float] = field(default_factory=list)   # steps 8+   (from CHILLER_OFFLINE_STEP=8)

    # ── Carbon tracking ───────────────────────────────────────────────────────
    carbon_cost_total: float           = 0.0
    high_carbon_cooling_kw: List[float] = field(default_factory=list)  # proxy values

    # ── Recovery window ───────────────────────────────────────────────────────
    # Steps 8–16 (CHILLER_OFFLINE_STEP to RECOVERY_WINDOW_END): track how many of these steps have all critical zones in safe band
    recovery_steps_total:  int = 0
    recovery_steps_safe:   int = 0

    # ── Triage quality ────────────────────────────────────────────────────────
    # After chiller failure, does agent protect critical zones at expense of low-priority?
    triage_quality_scores: List[float] = field(default_factory=list)

    # ── Reasoning coherence ───────────────────────────────────────────────────
    reasoning_coherence_scores: List[float] = field(default_factory=list)

    # ── Internal ──────────────────────────────────────────────────────────────
    _pid_baseline_pue: float = 1.55
    _last_step_num: int = -1

    def step(self, grader_input: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute step reward from grader_input dict.

        Expected keys (set by environment._build_grader_input):
          step                        : int
          zones                       : List[dict]
          current_pue                 : float
          pid_baseline_pue            : float
          carbon_intensity_normalized : float
          carbon_intensity_label      : str
          chiller_active              : bool
          chiller_setpoint_c          : float
          sla_violation_streak        : int
          action                      : DCAction  (Pydantic model)
          last_action                 : SimDCAction duck-type
          action_clipped              : dict
          reasoning                   : Optional[str]
        """
        # ── Unpack ────────────────────────────────────────────────────────────
        step_num: int          = grader_input.get("step", 0)
        current_pue: float     = grader_input["current_pue"]
        pid_baseline_pue: float = grader_input.get("pid_baseline_pue", self._pid_baseline_pue)
        self._pid_baseline_pue  = pid_baseline_pue
        carbon: float          = grader_input.get("carbon_intensity_normalized", 0.5)
        carbon_label: str      = grader_input.get("carbon_intensity_label", "medium")
        zones: List[dict]      = grader_input.get("zones", [])
        action                 = grader_input.get("action")
        last_action            = grader_input.get("last_action")
        chiller_active: bool   = grader_input.get("chiller_active", True)
        reasoning: Optional[str] = grader_input.get("reasoning")

        self.steps_total += 1
        self._last_step_num = step_num

        is_post_fault    = step_num >= CHILLER_OFFLINE_STEP
        is_recovery_window = CHILLER_OFFLINE_STEP <= step_num <= RECOVERY_WINDOW_END
        is_pre_fault     = step_num < FAULT_START_STEP
        is_high_carbon   = carbon >= HIGH_CARBON_THRESHOLD

        # ── PUE tracking by phase ─────────────────────────────────────────────
        if is_pre_fault:
            self.pre_fault_pue.append(current_pue)
        elif is_post_fault:
            self.post_fault_pue.append(current_pue)

        # ── Per-zone temperature reward + safety penalty ───────────────────────
        temp_reward_total  = 0.0
        safety_penalty     = 0.0
        all_critical_safe  = True
        n_zones            = max(len(zones), 1)

        for zone in zones:
            zone_id: str    = zone.get("zone_id", "")
            zone_temp: float = zone["temp_c"]
            priority: int    = zone.get("zone_priority", 1)
            consec_viol: int = zone.get("consecutive_violation", 0)
            is_critical      = (priority == 2)

            in_safe_band = TEMP_MIN <= zone_temp <= TEMP_MAX

            if in_safe_band:
                closeness = 1.0 - abs(zone_temp - TEMP_IDEAL) / 5.0
                base = 0.5 + 0.3 * closeness
                priority_mult = {0: 0.7, 1: 1.0, 2: 1.4}[priority]
                temp_reward_total += base * priority_mult
            else:
                violation = max(zone_temp - TEMP_MAX, TEMP_MIN - zone_temp)
                priority_mult = {0: 1.0, 1: 1.5, 2: 2.5}[priority]
                temp_reward_total += -0.4 * min(violation / 5.0, 1.0) * priority_mult

            # Hard safety penalty for critical zones above 30°C
            if is_critical and zone_temp > CRITICAL_THRESHOLD:
                all_critical_safe = False
                self.critical_zone_violations += 1

                if zone_temp > EMERGENCY_THRESHOLD:
                    safety_penalty += -1.0
                else:
                    streak_pen = 0.1 * min(consec_viol, 5)
                    safety_penalty += (-0.5 - streak_pen) * 2.0   # priority 2 multiplier

        temp_reward = STEP_TEMP_WEIGHT * (temp_reward_total / n_zones)
        safety_reward = STEP_SAFETY_WEIGHT * max(safety_penalty, -1.0)

        if all_critical_safe:
            self.steps_critical_safe += 1

        # ── Recovery window tracking ───────────────────────────────────────────
        if is_recovery_window:
            self.recovery_steps_total += 1
            if all_critical_safe:
                self.recovery_steps_safe += 1

        # ── PUE component ─────────────────────────────────────────────────────
        # Suppressed during chiller fault window or when any critical zone is above 25 °C:
        # high fans are mandatory then — penalising PUE discourages necessary cooling.
        denominator = max(pid_baseline_pue - IDEAL_PUE, 0.01)
        pue_improvement = (pid_baseline_pue - current_pue) / denominator
        pue_improvement = max(-0.5, min(1.0, pue_improvement))
        _any_critical_hot = any(
            z.get("temp_c", 22.0) > 25.0
            for z in zones
            if z.get("zone_id", "") in CRITICAL_ZONE_IDS
        )
        _pue_suppressed = is_post_fault or _any_critical_hot
        pue_reward = STEP_PUE_WEIGHT * pue_improvement if not _pue_suppressed else 0.0

        # ── Carbon component ──────────────────────────────────────────────────
        cooling_proxy = sum(z.get("fan_speed_pct", 50.0) / 100.0 for z in zones) / n_zones
        carbon_cost = cooling_proxy * carbon
        self.carbon_cost_total += carbon_cost
        if is_high_carbon:
            self.high_carbon_cooling_kw.append(cooling_proxy)
        carbon_reward = -STEP_CARBON_WEIGHT * carbon_cost

        # ── Roughness penalty ─────────────────────────────────────────────────
        roughness = _compute_roughness(action, last_action)
        roughness_reward = -STEP_ROUGH_WEIGHT * roughness

        # ── Stability bonus ───────────────────────────────────────────────────
        avg_consec_safe = (
            sum(z.get("consecutive_safe", 0) for z in zones) / n_zones
        )
        stability_bonus = STEP_STABILITY_WEIGHT * min(avg_consec_safe / 10.0, 1.0)

        # ── Triage quality (post-fault only) ──────────────────────────────────
        triage_score: Optional[float] = None
        if is_post_fault and zones:
            triage_score = _compute_triage_quality(zones, action)
            if triage_score is not None:
                self.triage_quality_scores.append(triage_score)

        # ── Reasoning coherence ───────────────────────────────────────────────
        coherence: Optional[float] = None
        if reasoning:
            coherence = score_reasoning_coherence(
                reasoning=reasoning,
                action=action,
                zones=zones,
                step_num=step_num,
                chiller_active=chiller_active,
                carbon_label=carbon_label,
                is_post_fault=is_post_fault,
            )
            self.reasoning_coherence_scores.append(coherence)

        # ── Combine ───────────────────────────────────────────────────────────
        total = round(
            temp_reward
            + pue_reward
            + carbon_reward
            + roughness_reward
            + stability_bonus
            + safety_reward,
            4,
        )
        total = max(-1.0, min(1.0, total))

        detail: Dict[str, Any] = {
            "temp_reward":         round(temp_reward, 4),
            "pue_reward":          round(pue_reward, 4),
            "carbon_reward":       round(carbon_reward, 4),
            "roughness_penalty":   round(roughness_reward, 4),
            "stability_bonus":     round(stability_bonus, 4),
            "safety_penalty":      round(safety_reward, 4),
            "all_critical_safe":   all_critical_safe,
            "is_recovery_window":  is_recovery_window,
            "is_high_carbon":      is_high_carbon,
            "pue":                 round(current_pue, 4),
            "pid_baseline_pue":    round(pid_baseline_pue, 4),
            "pue_improvement":     round(pue_improvement, 4),
            "triage_score":        round(triage_score, 4) if triage_score is not None else None,
            "reasoning_coherence": round(coherence, 4) if coherence is not None else None,
        }
        return total, detail

    def final_score(self) -> float:
        """
        Final episode score in [0.0, 1.0].

        Returns 0.0 immediately if the episode was SLA-terminated early
        (environment sets sla_terminated_early before calling this).

        Breakdown:
          30% — SLA compliance (critical zones safe fraction)
          25% — Carbon efficiency during high-carbon windows
          20% — Recovery speed after chiller failure
          15% — Triage quality
          10% — Reasoning coherence
        """
        if self.sla_terminated_early:
            return 0.0

        if self.steps_total == 0:
            return 0.0

        # ── SLA compliance ────────────────────────────────────────────────────
        sla_score = self.steps_critical_safe / self.steps_total

        # ── Carbon efficiency ─────────────────────────────────────────────────
        # Lower average cooling proxy during high-carbon windows = better score.
        # Score 1.0 if agent did zero high-carbon cooling; 0.0 if always at 100%.
        if self.high_carbon_cooling_kw:
            avg_high_carbon_cooling = sum(self.high_carbon_cooling_kw) / len(self.high_carbon_cooling_kw)
            carbon_score = max(0.0, 1.0 - avg_high_carbon_cooling)
        else:
            carbon_score = 0.5   # episode ended before any high-carbon window — neutral credit

        # ── Recovery speed ────────────────────────────────────────────────────
        if self.recovery_steps_total > 0:
            recovery_score = self.recovery_steps_safe / self.recovery_steps_total
        else:
            recovery_score = sla_score   # no post-fault window observed → use SLA

        # ── Triage quality ────────────────────────────────────────────────────
        if self.triage_quality_scores:
            triage_score = sum(self.triage_quality_scores) / len(self.triage_quality_scores)
        else:
            triage_score = 0.5   # no post-fault steps → neutral

        # ── Reasoning coherence ───────────────────────────────────────────────
        if self.reasoning_coherence_scores:
            reasoning_score = sum(self.reasoning_coherence_scores) / len(self.reasoning_coherence_scores)
        else:
            reasoning_score = 0.0   # agent never provided reasoning → penalise

        score = (
            FINAL_SLA_WEIGHT       * sla_score
            + FINAL_CARBON_WEIGHT  * carbon_score
            + FINAL_RECOVERY_WEIGHT * recovery_score
            + FINAL_TRIAGE_WEIGHT  * triage_score
            + FINAL_REASONING_WEIGHT * reasoning_score
        )
        return round(score, 4)

    def mark_sla_terminated(self):
        """Called by environment when a critical zone triggers hard termination."""
        self.sla_terminated_early = True


# ── Triage quality ─────────────────────────────────────────────────────────────

def _compute_triage_quality(zones: List[dict], action: Any) -> Optional[float]:
    """
    Measure how well the agent triages after chiller failure.

    Good triage: critical zones get higher fan speed than sacrifice zones.
    Perfect score (1.0): every critical zone has higher fan than every sacrifice zone.
    Zero score (0.0): every sacrifice zone has equal or higher fan than critical zones.

    Returns None if zones don't include both critical and sacrifice zone types.
    """
    if action is None:
        return None

    action_fan_map: Dict[str, float] = {
        getattr(adj, "zone_id", ""): getattr(adj, "fan_speed_pct", 50.0)
        for adj in getattr(action, "zone_adjustments", [])
    }

    critical_fans  = [action_fan_map[zid] for zid in CRITICAL_ZONE_IDS if zid in action_fan_map]
    sacrifice_fans = [action_fan_map[zid] for zid in SACRIFICE_ZONE_IDS if zid in action_fan_map]

    if not critical_fans or not sacrifice_fans:
        return None   # can't assess triage without both zone types present

    avg_critical  = sum(critical_fans) / len(critical_fans)
    avg_sacrifice = sum(sacrifice_fans) / len(sacrifice_fans)

    # Score = how much more airflow critical zones get vs sacrifice zones.
    # Capped: difference of 30+ fan points = full triage credit.
    fan_advantage = avg_critical - avg_sacrifice
    return round(max(0.0, min(1.0, fan_advantage / 30.0)), 4)


# ── Roughness ─────────────────────────────────────────────────────────────────

def _compute_roughness(action: Any, last_action: Any) -> float:
    """Normalised action roughness [0–1]. Returns 0 if either action is absent."""
    if action is None or last_action is None:
        return 0.0

    try:
        chiller_delta = abs(action.chiller_setpoint_c - last_action.chiller_setpoint_c)
    except AttributeError:
        chiller_delta = 0.0

    last_zone_map = {
        adj.zone_id: adj
        for adj in getattr(last_action, "zone_adjustments", [])
    }
    fan_deltas, supply_deltas = [], []
    for adj in getattr(action, "zone_adjustments", []):
        last_adj = last_zone_map.get(adj.zone_id)
        if last_adj is None:
            continue
        fan_deltas.append(abs(adj.fan_speed_pct - last_adj.fan_speed_pct))
        supply_deltas.append(abs(adj.supply_air_temp_setpoint_c - last_adj.supply_air_temp_setpoint_c))

    avg_fan    = sum(fan_deltas) / max(len(fan_deltas), 1)
    avg_supply = sum(supply_deltas) / max(len(supply_deltas), 1)

    return min(
        (avg_fan / 100.0) * 0.50
        + (avg_supply / 10.0) * 0.30
        + (chiller_delta / 9.0) * 0.20,
        1.0,
    )


# ── Reasoning coherence ────────────────────────────────────────────────────────

def score_reasoning_coherence(
    reasoning: str,
    action: Any,
    zones: List[dict],
    step_num: int,
    chiller_active: bool,
    carbon_label: str,
    is_post_fault: bool,
) -> float:
    """
    Heuristic coherence check: is the agent's stated reasoning consistent with
    its actual action?

    Returns a score in [0.0, 1.0].
      1.0 = no detectable incoherence
      0.0 = clear contradiction between reasoning and action

    Checks performed (each incoherence subtracts from 1.0):
      1. Fan direction claims: says "increasing fan for zone X" but sets it lower
      2. Carbon claims: says "reducing cooling" during high carbon but increases fans
      3. Chiller failure claims: says "chiller failed" before fault step
      4. Triage claims: says "protecting critical zones" but sets sacrifice fans higher
      5. Temperature claims: says zone is "overheating" / "stable" inconsistently
    """
    if not reasoning or not action:
        return 0.0

    reasoning_lower = reasoning.lower()
    penalty = 0.0
    checks  = 0

    action_fan_map: Dict[str, float] = {
        getattr(adj, "zone_id", ""): getattr(adj, "fan_speed_pct", 50.0)
        for adj in getattr(action, "zone_adjustments", [])
    }
    zone_temp_map: Dict[str, float] = {
        z.get("zone_id", ""): z.get("temp_c", 22.0)
        for z in zones
    }

    # ── Check 1: Fan direction claims ─────────────────────────────────────────
    # "increasing fan" / "raising airflow" for a specific zone
    for zone_id, fan_pct in action_fan_map.items():
        zone_short = zone_id.replace("_", " ")   # e.g. "zone ai 1"
        if _mentions_zone(reasoning_lower, zone_id):
            if _says_increasing_fan(reasoning_lower, zone_id) and fan_pct < 50.0:
                penalty += 0.25
                checks  += 1
            elif _says_decreasing_fan(reasoning_lower, zone_id) and fan_pct > 80.0:
                penalty += 0.15
                checks  += 1

    # ── Check 2: Carbon / cooling reduction claims ────────────────────────────
    checks += 1
    if carbon_label in ("high", "critical_high"):
        claims_reduction = any(
            phrase in reasoning_lower
            for phrase in ("reducing cooling", "less cooling", "lower fan", "cut fan",
                           "carbon", "dirty grid", "efficiency")
        )
        actual_avg_fan = (
            sum(action_fan_map.values()) / max(len(action_fan_map), 1)
        )
        if claims_reduction and actual_avg_fan > 85.0:
            penalty += 0.20   # claims reduction but maxed all fans
        elif not claims_reduction and actual_avg_fan < 30.0 and not is_post_fault:
            penalty += 0.10   # no mention of carbon but suspiciously low fans during high carbon

    # ── Check 3: Premature chiller failure claims ─────────────────────────────
    checks += 1
    mentions_chiller_failure = any(
        phrase in reasoning_lower
        for phrase in ("chiller failed", "chiller offline", "chiller down",
                       "no chiller", "chiller fault", "chiller gone")
    )
    if mentions_chiller_failure and step_num < FAULT_START_STEP:
        penalty += 0.30   # claimed failure before it happened — hallucination

    # ── Check 4: Triage claims ────────────────────────────────────────────────
    checks += 1
    claims_triage = any(
        phrase in reasoning_lower
        for phrase in ("protecting critical", "priorit", "sacrific", "triage",
                       "zone_infra", "zone_storage", "lower priority")
    )
    if claims_triage and is_post_fault:
        critical_fans  = [action_fan_map[z] for z in CRITICAL_ZONE_IDS if z in action_fan_map]
        sacrifice_fans = [action_fan_map[z] for z in SACRIFICE_ZONE_IDS if z in action_fan_map]
        if critical_fans and sacrifice_fans:
            avg_crit = sum(critical_fans) / len(critical_fans)
            avg_sac  = sum(sacrifice_fans) / len(sacrifice_fans)
            if avg_sac > avg_crit + 15.0:
                # Claims triage but sacrifice zones actually have more airflow
                penalty += 0.25

    # ── Check 5: Temperature state claims ────────────────────────────────────
    # "zone_ai_1 is overheating / too hot" — verify it actually is
    for zone_id, zone_temp in zone_temp_map.items():
        if not _mentions_zone(reasoning_lower, zone_id):
            continue
        checks += 1
        if _says_zone_overheating(reasoning_lower, zone_id) and zone_temp < 26.0:
            penalty += 0.15   # claims overheating but zone is comfortably cool
        if _says_zone_stable(reasoning_lower, zone_id) and zone_temp > 29.0:
            penalty += 0.20   # claims stable but zone is near SLA limit

    total_penalty = min(penalty, 1.0)
    return round(max(0.0, 1.0 - total_penalty), 4)


# ── Reasoning keyword helpers ─────────────────────────────────────────────────

def _mentions_zone(text: str, zone_id: str) -> bool:
    """Return True if reasoning mentions this zone (handles both formats)."""
    return zone_id in text or zone_id.replace("_", " ") in text or zone_id.replace("_", "-") in text


def _says_increasing_fan(text: str, zone_id: str) -> bool:
    """Return True if text claims fan speed is being increased for this zone."""
    patterns = [
        r"increas\w* fan.{0,30}" + re.escape(zone_id.replace("_", r"\W*")),
        re.escape(zone_id.replace("_", r"\W*")) + r".{0,30}increas\w* fan",
        r"rais\w* airflow.{0,30}" + re.escape(zone_id.replace("_", r"\W*")),
        r"boost\w*.{0,20}" + re.escape(zone_id.replace("_", r"\W*")),
    ]
    zone_pattern = zone_id.replace("_", r"[_ -]?")
    return any(
        re.search(p.replace(re.escape(zone_id.replace("_", r"\W*")), zone_pattern), text)
        for p in patterns
    )


def _says_decreasing_fan(text: str, zone_id: str) -> bool:
    """Return True if text claims fan speed is being decreased for this zone."""
    zone_pattern = zone_id.replace("_", r"[_ -]?")
    patterns = [
        r"reduc\w* fan",
        r"lower\w* fan",
        r"decreas\w* fan",
        r"cut\w* fan",
        r"sacrific\w*.{0,30}" + zone_pattern,
    ]
    return any(re.search(p, text) for p in patterns)


def _says_zone_overheating(text: str, zone_id: str) -> bool:
    """Return True if text claims this zone is overheating."""
    zone_pattern = zone_id.replace("_", r"[_ -]?")
    overheat_terms = r"(overheat|too hot|above.{0,10}thresh|exceed|critical temp|danger)"
    return bool(re.search(zone_pattern + r".{0,40}" + overheat_terms, text)) or \
           bool(re.search(overheat_terms + r".{0,40}" + zone_pattern, text))


def _says_zone_stable(text: str, zone_id: str) -> bool:
    """Return True if text claims this zone is stable / within safe range."""
    zone_pattern = zone_id.replace("_", r"[_ -]?")
    stable_terms = r"(stable|in range|safe|nominal|within.{0,10}limit|no concern)"
    return bool(re.search(zone_pattern + r".{0,40}" + stable_terms, text)) or \
           bool(re.search(stable_terms + r".{0,40}" + zone_pattern, text))