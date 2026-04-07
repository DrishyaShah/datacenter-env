"""
inference.py — DC-OpenEnv Baseline Inference Script
====================================================
Runs the datacenter cooling environment against an LLM via an OpenAI-compatible API.

Environment variables:
    HF_TOKEN or OPENAI_API_KEY   API key (either is accepted)
    API_BASE_URL                 e.g. https://api.groq.com/openai/v1  (default: Groq)
    MODEL_NAME                   e.g. llama-3.3-70b-versatile  (default: 70b versatile)
    INFERENCE_MAX_STEPS_PER_TASK Optional cap per task (int) to stay under time limits
    VERBOSE                      If "1", print non-protocol [INFO]/[SCORE] lines

Stdout (hackathon protocol — reward/score/rewards use 2 decimal places):
    [START] task=<task> env=dc-openenv model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<str|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<csv>
"""

from __future__ import annotations

import os
import json
import sys
import textwrap
import time
from typing import List, Optional

from openai import OpenAI, RateLimitError

from server.environment import DCEnvironment
from server.models import DCAction, DCObservation, ZoneAdjustment


# ── Tee: dual-output logger ───────────────────────────────────────────────────

class Tee:
    """Write to multiple file objects simultaneously (stdout + log file)."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("OPENAI_API_KEY") or ""
VERBOSE      = os.getenv("VERBOSE", "").strip().lower() in ("1", "true", "yes")

_steps_cap_raw = os.getenv("INFERENCE_MAX_STEPS_PER_TASK", "").strip()
INFERENCE_MAX_STEPS_PER_TASK: Optional[int] = (
    int(_steps_cap_raw) if _steps_cap_raw.isdigit() else None
)

# OpenAI client — initialised in main() after key validation
client: Optional[OpenAI] = None

# Safe-band bounds — shared by alert generation and history tagging
TEMP_MAX = 27.0
TEMP_MIN = 18.0

# Per-task success thresholds calibrated to each difficulty level.
# Medium: 20% of score weight is always dead (peak_score unreachable in 25 steps).
# Hard: cascading failure + carbon window is genuinely hard; set expectation accordingly.
SUCCESS_THRESHOLDS = {
    "easy-single-zone":       0.55,
    "medium-multi-zone":      0.50,
    "hard-cascading-failure": 0.40,
}

# ── Rate-limit / retry config ─────────────────────────────────────────────────
STEP_SLEEP_SECONDS   = 0            # no inter-step sleep — stay well under 20-min cap
LLM_MAX_RETRIES      = 3            # attempts before falling back to held settings
LLM_RETRY_BASE_SLEEP = 2.0         # seconds; doubles on each retry (2 → 4 → 8 = 14s max)
                                    # was 5.0 (5→10→20 = 35s max) which caused >20 min runs

# ── Global wall-clock budget ──────────────────────────────────────────────────
# Hard cap: stop starting new tasks if we are within GLOBAL_TIMEOUT_BUFFER seconds
# of the GLOBAL_TIMEOUT_SECONDS deadline.  Evaluator hard-kills at 20 minutes.
GLOBAL_TIMEOUT_SECONDS = 18 * 60   # 18 minutes — 2-min safety margin
GLOBAL_TIMEOUT_BUFFER  = 90        # seconds: don't start a task that can't finish
_SCRIPT_START: float = 0.0         # populated in main() via time.time()

# ── Task registry (all three tasks run against DCEnvironment + graders) ───────
TASKS = [
    {
        "name": "easy-single-zone",
        "description": "Single-zone thermal runaway recovery under steady load",
        "max_steps": 20,   # 20 steps × 12 min/step = 4 hr (full arc)
    },
    {
        "name": "medium-multi-zone",
        "description": "Multi-zone load surge with faulty sensor and diurnal variation",
        "max_steps": 30,   # 30 steps × 24 min/step = 12 hr (full 06:00–18:00 arc)
    },
    {
        "name": "hard-cascading-failure",
        "description": "Cascading chiller failure with carbon-aware triage",
        "max_steps": 40,   # 40 steps × 36 min/step = 24 hr (full arc)
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a data centre operations engineer AI managing server room cooling.

    === MDP STRUCTURE ===
    State : zone temperatures, fan speeds, supply setpoints, chiller state, PUE,
            carbon intensity, SLA streak, 3-step history buffer.
    Action: per-zone fan_speed_pct [0-100] + supply_air_temp_setpoint_c [16-26],
            chiller_setpoint_c [6-15], chiller_active bool.
    Reward: shaped by temperature compliance, PUE vs baseline, carbon cost, smoothness.
    Goal  : keep ALL zones in [18-27 °C], minimise PUE, prefer low-carbon cooling.

    === DECISION RULES (priority order) ===
    1. SAFETY FIRST — all zones must stay in [18, 27] °C at every step.
    2. EFFICIENCY — lower PUE is better; avoid high fans when the zone is already safe.
    3. CARBON — prefer lower fan speeds during high-carbon grid windows.

    At each step you receive the current data centre state as JSON.
    The "zones" array lists each zone with its zone_id and all sensor readings.

    You MUST respond with ONLY a valid JSON object — no markdown, no text outside JSON:
    {
      "zone_adjustments": [
        {
          "zone_id": "<exact zone_id copied from the observation zones array>",
          "fan_speed_pct": <number 0.0 to 100.0>,
          "supply_air_temp_setpoint_c": <number 16.0 to 26.0>
        }
      ],
      "chiller_setpoint_c": <number 6.0 to 15.0>,
      "chiller_active": <true or false>,
      "reasoning": "<one sentence explaining your main decision>"
    }

    === ZONE CONTROL RULES ===
    - Copy EXACT zone_id strings from the observation — never invent zone names.
    - Include ALL zones from the observation in zone_adjustments every step.
    - cold_aisle_temp_c ABOVE 27 °C → fan high (80-100), supply setpoint low (16-18).
    - cold_aisle_temp_c BELOW 18 °C → you are OVERCOOLING (same severity as overheating).
      Immediately raise supply_air_temp_setpoint_c by 2, reduce fan_speed_pct by 15-20.
    - cold_aisle_temp_c between 19-23 °C AND falling for 2+ steps → back off NOW.
      Set fan=45-55, supply=21-22 to avoid drifting below 18 °C.
    - Thermal inertia: the zone keeps cooling for 2-3 steps after you reduce fans.
      Back off BEFORE the zone hits 18 °C, not after.
    - Overheating recovery: aggressive cooling (fan=85-95) only while temp > 24 °C.
      Switch to moderate (fan=55-70) once temp drops below 24 °C.
    - Zone in safe range [18-27] → moderate fan (40-65), supply near 20-22 to save energy.

    === SENSOR CONFIDENCE RULE ===
    - Each zone has a sensor_confidence field [0.0-1.0].
    - If sensor_confidence < 0.5, the reported_temp_c is UNRELIABLE (sensor may be drifted
      or faulty — reporting up to 12 °C above the true temperature).
    - When sensor_confidence < 0.5, trust cold_aisle_temp_c instead of reported_temp_c.
    - Never set max fans based solely on a high reported_temp_c when sensor_confidence < 0.5.
      Instead use cold_aisle_temp_c and hot_aisle_temp_c to judge the true thermal state.

    === CHILLER FAILURE PROTOCOL (HIGHEST PRIORITY AFTER SAFETY) ===
    If chiller_fault_detected is True OR chiller_cop < 2.0:
      - You have 5-10 steps before the chiller goes offline.
      - IMMEDIATELY: set ALL zone fans to 80-90%, supply setpoints to 18-19 °C.
      - Do NOT wait for temperatures to rise before acting. Pre-cool now.
      - Continue running chiller (chiller_active: true) — it still helps even degraded.
      - Do NOT raise chiller_setpoint_c above 10 during a fault — keep the water cold.

    If chiller_active is False (chiller is offline):
      - Fans are your ONLY cooling — chiller provides zero cooling capacity.
      - Set CRITICAL zones (zone_priority=2) to fan=90-100%, supply=16-18 °C.
      - Accept LOW-priority zones (zone_priority=0) drifting toward 26-27 °C to
        concentrate cooling capacity on CRITICAL zones.
      - Carbon cost is irrelevant during a chiller failure — survival takes priority.
      - Do NOT set chiller_active: true — it will not respond while offline.
      - Check the "alerts" field each step; it will tell you when the chiller comes back.

    If you see "[CHILLER_FAULT]" or "[CHILLER_OFFLINE]" tags in the step history:
      - The failure has been ongoing. Review how many steps it has been active.
      - If 3+ steps of OFFLINE: assume CRITICAL zones need sustained max cooling.
      - Gradually scale back ONLY once you see temperatures stabilising below 24 °C.

    === CHILLER RULES ===
    - Keep chiller_active true unless you have a specific reason to disable it.
    - chiller_setpoint_c MUST be between 6.0 and 15.0 — never go above 15.
      Values above 15 will be silently clamped to 15.
    - A reasonable default is chiller_setpoint_c = 10.0 for most situations.
    - Lower chiller_setpoint_c (e.g. 8.0) = colder water = more cooling power but
      higher energy cost. Use 8-10 for recovery, 11-14 for steady efficient operation.
    - If chiller_fault_detected is true: chiller may be offline or degraded.
      Compensate with higher fan speeds and lower supply setpoints.

    === TRIAGE RULE (multi-zone only) ===
    - Zones have zone_priority: 2=CRITICAL, 1=MEDIUM, 0=LOW.
    - If cooling capacity is constrained, protect CRITICAL zones first.
    - It is acceptable (even correct) to allow LOW-priority zones to drift warm
      temporarily if doing so keeps CRITICAL zones safe.

    === RATE LIMITS (physics constraint) ===
    - Fan speed can only change ±20 % per step.
    - Supply air setpoint can only change ±2 °C per step.
    - Chiller setpoint can only change ±1 °C per step.
    - Plan ahead — you cannot jump from fan=40 to fan=100 in one step.
""").strip()


# ── Logging helpers (2 decimal places per hackathon sample) ──────────────────
_logged_steps: set = set()


def log_start(task: str, model: str) -> None:
    global _logged_steps
    _logged_steps = set()
    print(f"[START] task={task} env=dc-openenv model={model}", flush=True)


def log_step(step: int, action_json: str, reward: float, done: bool, error: Optional[str]) -> None:
    if step in _logged_steps:
        return
    _logged_steps.add(step)
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_json} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_csv}",
        flush=True,
    )


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


# ── Runtime alert generation ──────────────────────────────────────────────────
def _compute_alerts(obs_dict: dict, prev_temps: dict) -> List[str]:
    """
    Derive human-readable alert strings from the current observation.

    These are injected into the LLM's JSON observation so it can react to
    critical events (chiller failure, temperature violations, sensor faults,
    rising trends) without having to infer them from raw numbers alone.
    """
    alerts: List[str] = []
    zones = obs_dict.get("zones", [])

    # ── Chiller state ─────────────────────────────────────────────────────────
    if not obs_dict.get("chiller_active", True):
        alerts.append(
            "CRITICAL: Chiller is OFFLINE — fans are your ONLY cooling tool. "
            "Set all zone fans to 90-100% immediately, especially CRITICAL zones."
        )
    elif obs_dict.get("chiller_fault_detected", False):
        cop = obs_dict.get("chiller_cop", 3.5)
        alerts.append(
            f"WARNING: Chiller fault detected (current COP={cop:.2f}, nominal 3.5). "
            "Chiller may go offline within 5-10 steps. Ramp ALL fans up now."
        )

    # ── Per-zone alerts ───────────────────────────────────────────────────────
    for z in zones:
        zid  = z["zone_id"]
        temp = z.get("cold_aisle_temp_c", 22.0)
        conf = z.get("sensor_confidence", 1.0)
        rep  = z.get("reported_temp_c", temp)
        prev = prev_temps.get(zid, temp)
        delta = temp - prev

        # Approaching boundary (1°C margin)
        if TEMP_MAX - 1.0 < temp <= TEMP_MAX:
            alerts.append(
                f"WARNING: {zid} at {temp:.1f}°C — within 1°C of violation limit "
                f"({TEMP_MAX}°C). Increase cooling now."
            )
        elif TEMP_MIN <= temp < TEMP_MIN + 1.0:
            alerts.append(
                f"WARNING: {zid} at {temp:.1f}°C — within 1°C of overcooling limit "
                f"({TEMP_MIN}°C). Reduce fan speed or raise supply setpoint."
            )

        # Active violation
        if temp > TEMP_MAX:
            alerts.append(
                f"VIOLATION: {zid} is OVERHEATING at {temp:.1f}°C. "
                "Immediate max cooling required."
            )
        elif temp < TEMP_MIN:
            alerts.append(
                f"VIOLATION: {zid} is OVERCOOLING at {temp:.1f}°C. "
                "Raise supply_air_temp_setpoint_c by 2°C and cut fans 15-20%."
            )

        # Faulty sensor
        if conf < 0.5:
            bias = round(rep - temp, 1)
            alerts.append(
                f"SENSOR FAULT: {zid} sensor_confidence={conf:.2f} — reported_temp "
                f"({rep:.1f}°C) is approximately {bias:+.1f}°C off from actual "
                f"cold_aisle_temp_c ({temp:.1f}°C). "
                "DO NOT set fans based on reported_temp. Use cold_aisle_temp_c."
            )

        # Rising fast and already warm
        if delta > 0.8 and temp > 23.0:
            alerts.append(
                f"TREND: {zid} rising fast (+{delta:.1f}°C this step, now {temp:.1f}°C). "
                "Act now — thermal inertia means it will keep rising 2-3 more steps."
            )

        # Zone is safe, temperature is stable, but fans are still running high —
        # signal the LLM to back off for PUE efficiency.
        fan = z.get("fan_speed_pct", 50.0)
        if (
            TEMP_MIN + 0.5 < temp < TEMP_MAX - 2.0  # safely within bounds
            and abs(delta) < 0.3                      # temperature not changing
            and fan > 70.0                            # fans are wasteful
        ):
            alerts.append(
                f"EFFICIENCY: {zid} is stable at {temp:.1f}°C with fan at {fan:.0f}% "
                "— this is wasteful. Reduce fan to 45-65% to improve PUE. "
                "Thermal inertia will sustain current cooling for 2-3 more steps."
            )

    # ── Carbon ────────────────────────────────────────────────────────────────
    carbon = obs_dict.get("carbon_intensity_normalized", 0.5)
    if carbon > 0.8:
        alerts.append(
            f"CARBON CRITICAL ({carbon:.2f}): Grid is at peak emissions. "
            "Reduce fan speeds where zone temps allow — every percent matters."
        )

    # ── SLA streak ────────────────────────────────────────────────────────────
    streak = obs_dict.get("sla_violation_streak", 0)
    if streak >= 5:
        alerts.append(
            f"SLA ALERT: {streak} consecutive violation steps detected. "
            "Hard termination triggers at 10. Take urgent corrective action NOW."
        )

    return alerts


# ── LLM call with retry on 429 ────────────────────────────────────────────────
def get_llm_action(obs_dict: dict, step: int, history: List[str]) -> dict:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Step {step} — Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent history (oldest → newest):
        {history_block}

        Respond with your JSON action. Use the exact zone_id values shown above.
    """).strip()

    last_exc: Exception = RuntimeError("no attempts made")

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                raise ValueError("empty model response")

            if "```" in raw:
                parts = raw.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        raw = part
                        break

            return json.loads(raw.strip())

        except RateLimitError as e:
            last_exc = e
            err_msg = str(e).lower()
            # "tokens per day" quota is exhausted for today — retrying will never help.
            # Skip all remaining attempts and let the caller use the last intended fallback.
            if "per day" in err_msg or "tokens per day" in err_msg or "tpd" in err_msg:
                _vprint(
                    f"[WARN] Daily token quota exhausted at step {step} "
                    "— skipping retries, using fallback action"
                )
                return {}
            # Normal per-minute rate limit — exponential backoff and retry
            sleep_for = LLM_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            _vprint(
                f"[WARN] 429 rate-limit on attempt {attempt}/{LLM_MAX_RETRIES} "
                f"at step {step} — sleeping {sleep_for:.0f}s before retry"
            )
            time.sleep(sleep_for)

    # All retries exhausted — return empty dict so caller falls back gracefully
    _vprint(f"[WARN] All {LLM_MAX_RETRIES} LLM attempts failed at step {step}, using fallback")
    return {}


# ── Build DCAction from LLM output ────────────────────────────────────────────
def build_action(llm_result: dict, obs: DCObservation) -> DCAction:
    zone_obs_map = {z.zone_id: z for z in obs.zones}
    zone_ids = list(zone_obs_map.keys())

    adjustments_raw = llm_result.get("zone_adjustments", [])
    adjustments = []

    for i, adj in enumerate(adjustments_raw):
        llm_zone_id = adj.get("zone_id", "")
        fan_speed    = float(adj.get("fan_speed_pct", 80.0))
        supply_temp  = float(adj.get("supply_air_temp_setpoint_c", 20.0))

        if llm_zone_id in zone_obs_map:
            actual_zone_id = llm_zone_id
        elif i < len(zone_ids):
            actual_zone_id = zone_ids[i]
        else:
            continue

        adjustments.append(
            ZoneAdjustment(
                zone_id=actual_zone_id,
                fan_speed_pct=max(0.0, min(100.0, fan_speed)),
                supply_air_temp_setpoint_c=max(16.0, min(26.0, supply_temp)),
            )
        )

    covered = {a.zone_id for a in adjustments}
    for zone_id in zone_ids:
        if zone_id not in covered:
            z = zone_obs_map[zone_id]
            adjustments.append(
                ZoneAdjustment(
                    zone_id=zone_id,
                    fan_speed_pct=z.fan_speed_pct,
                    supply_air_temp_setpoint_c=z.supply_air_temp_setpoint_c,
                )
            )

    return DCAction(
        zone_adjustments=adjustments,
        chiller_setpoint_c=max(6.0, min(15.0, float(llm_result.get("chiller_setpoint_c", 10.0)))),
        chiller_active=bool(llm_result.get("chiller_active", True)),
        reasoning=llm_result.get("reasoning", ""),
    )


def _effective_max_steps(task_max: int) -> int:
    if INFERENCE_MAX_STEPS_PER_TASK is not None:
        return min(INFERENCE_MAX_STEPS_PER_TASK, task_max)
    return task_max


# ── Run one task episode ───────────────────────────────────────────────────────
def run_task(task_cfg: dict) -> float:
    task_name = task_cfg["name"]
    max_steps = _effective_max_steps(int(task_cfg["max_steps"]))

    log_start(task=task_name, model=MODEL_NAME)

    env = DCEnvironment(task=task_name)
    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    score   = 0.0
    success = False

    try:
        obs: DCObservation = env.reset()

        _prev_temps: dict = {}
        _last_llm_result: dict = {}   # last non-empty LLM response — used as fallback intent

        for step in range(1, max_steps + 1):
            obs_dict = {
                "step": obs.step,
                "timestamp_hour": obs.timestamp_hour,
                "outside_temp_c": obs.outside_temp_c,
                "wet_bulb_temp_c": obs.wet_bulb_temp_c,
                "current_pue": obs.current_pue,
                "chiller_active": obs.chiller_active,
                "chiller_setpoint_c": obs.chiller_setpoint_c,
                "chiller_cop": obs.chiller_cop,
                "chiller_fault_detected": obs.chiller_fault_detected,
                "grid_carbon_intensity": obs.grid_carbon_intensity,
                "carbon_intensity_normalized": obs.carbon_intensity_normalized,
                "load_curve_phase": obs.load_curve_phase,
                "sla_violation_streak": obs.sla_violation_streak,
                "zones": [
                    {
                        "zone_id": z.zone_id,
                        "cold_aisle_temp_c": z.cold_aisle_temp_c,
                        "hot_aisle_temp_c": z.hot_aisle_temp_c,
                        "reported_temp_c": z.reported_temp_c,
                        "fan_speed_pct": z.fan_speed_pct,
                        "supply_air_temp_c": z.supply_air_temp_c,
                        "supply_air_temp_setpoint_c": z.supply_air_temp_setpoint_c,
                        "it_load_kw": z.it_load_kw,
                        "it_load_pct": z.it_load_pct,
                        "humidity_pct": z.humidity_pct,
                        "sensor_confidence": z.sensor_confidence,
                        "zone_priority": z.zone_priority,
                        "load_forecast_next_hour": z.load_forecast_next_hour,
                    }
                    for z in obs.zones
                ],
                "maintenance_notes": obs.maintenance_notes,
                "upcoming_events": obs.upcoming_events,
            }

            # Inject runtime alerts so the LLM can react to critical events explicitly
            alerts = _compute_alerts(obs_dict, _prev_temps)
            if alerts:
                obs_dict["alerts"] = alerts

            error_str = None
            try:
                llm_result = get_llm_action(obs_dict, step, history)
                if llm_result:
                    _last_llm_result = llm_result   # store last successful intent
                action = build_action(llm_result if llm_result else _last_llm_result, obs)
            except Exception as e:
                error_str = str(e)[:120]
                _vprint(f"[WARN] step {step} LLM failed ({error_str}), holding last intended action")
                if _last_llm_result:
                    action = build_action(_last_llm_result, obs)
                else:
                    # No prior LLM intent yet — use safe conservative defaults
                    action = DCAction(
                        zone_adjustments=[
                            ZoneAdjustment(
                                zone_id=z.zone_id,
                                fan_speed_pct=70.0,
                                supply_air_temp_setpoint_c=20.0,
                            )
                            for z in obs.zones
                        ],
                        chiller_setpoint_c=10.0,
                        chiller_active=True,
                        reasoning="fallback: LLM unavailable — safe defaults (no prior intent)",
                    )

            action_json = json.dumps(
                {
                    "zone_adjustments": [a.model_dump() for a in action.zone_adjustments],
                    "chiller_setpoint_c": action.chiller_setpoint_c,
                    "chiller_active": action.chiller_active,
                },
                separators=(",", ":"),
            )

            done = False
            try:
                obs    = env.step(action)
                reward = float(obs.reward) if obs.reward is not None else 0.0
                done   = obs.done
            except Exception as e:
                reward    = 0.0
                done      = True
                error_str = str(e)[:120]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action_json=action_json,
                reward=reward,
                done=done,
                error=error_str,
            )

            # Build zone summary and update _prev_temps with post-step values
            zone_parts = []
            for z in obs.zones:
                prev  = _prev_temps.get(z.zone_id, z.cold_aisle_temp_c)
                delta = z.cold_aisle_temp_c - prev
                _prev_temps[z.zone_id] = z.cold_aisle_temp_c
                zone_parts.append(
                    f"{z.zone_id}={z.cold_aisle_temp_c:.1f}C({delta:+.1f}) "
                    f"fan={z.fan_speed_pct:.0f}% supply={z.supply_air_temp_setpoint_c:.0f}C"
                )

            # Collect event tags so the LLM can spot fault/violation history at a glance
            tags = []
            if obs.chiller_fault_detected:
                tags.append("[CHILLER_FAULT]")
            if not obs.chiller_active:
                tags.append("[CHILLER_OFFLINE]")
            for z in obs.zones:
                if z.cold_aisle_temp_c > TEMP_MAX or z.cold_aisle_temp_c < TEMP_MIN:
                    tags.append(f"[VIOLATION:{z.zone_id}]")
                if z.sensor_confidence < 0.5:
                    tags.append(f"[SENSOR_BAD:{z.zone_id}]")
            tag_str = " ".join(tags)

            history.append(
                f"Step {step}{(' ' + tag_str) if tag_str else ''}: "
                f"{', '.join(zone_parts)} | "
                f"pue={obs.current_pue:.3f} | carbon={obs.grid_carbon_intensity} | "
                f"reward={reward:.2f}"
            )

            if done:
                break

            # Per-step wall-clock guard: stop the episode if we are running out of time
            elapsed = time.time() - _SCRIPT_START
            if elapsed >= GLOBAL_TIMEOUT_SECONDS - GLOBAL_TIMEOUT_BUFFER:
                _vprint(f"[WARN] Wall-clock budget exhausted at step {step} — ending episode early")
                break

            if STEP_SLEEP_SECONDS > 0:
                time.sleep(STEP_SLEEP_SECONDS)

        grader = getattr(env, "_grader", None)
        if grader is not None and hasattr(grader, "final_score"):
            score = float(grader.final_score())
        elif rewards:
            score = max(0.0, min(1.0, (sum(rewards) / len(rewards) + 1.0) / 2.0))

        success = score >= SUCCESS_THRESHOLDS.get(task_name, 0.55)

    except Exception as e:
        _vprint(f"[DEBUG] Fatal error in task {task_name}: {e}")

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    global client, _SCRIPT_START
    _SCRIPT_START = time.time()

    # ── Set up dual-output logging (stdout + file) ────────────────────────────
    log_file = open("inference_output.txt", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    # ── Validate API key ──────────────────────────────────────────────────────
    if not API_KEY:
        raise RuntimeError(
            "Set HF_TOKEN or OPENAI_API_KEY.\n"
            "  PowerShell: $env:HF_TOKEN='...'\n"
            "  bash:       export HF_TOKEN='...'"
        )

    # ── Initialise OpenAI-compatible client ───────────────────────────────────
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    _vprint(f"[INFO] API_BASE_URL = {API_BASE_URL}")
    _vprint(f"[INFO] MODEL_NAME   = {MODEL_NAME}")
    _vprint(f"[INFO] Running {len(TASKS)} tasks")

    all_scores = []
    for task_cfg in TASKS:
        elapsed = time.time() - _SCRIPT_START
        remaining = GLOBAL_TIMEOUT_SECONDS - elapsed
        if remaining < GLOBAL_TIMEOUT_BUFFER:
            _vprint(
                f"[WARN] Skipping task '{task_cfg['name']}' — only {remaining:.0f}s "
                f"remaining (need >{GLOBAL_TIMEOUT_BUFFER}s buffer)"
            )
            all_scores.append(0.0)
            continue
        score = run_task(task_cfg)
        all_scores.append(score)
        _vprint(f"[SCORE] {task_cfg['name']} => {score:.2f}")

    overall = sum(all_scores) / len(all_scores)
    _vprint(f"[FINAL] overall_score={overall:.2f}")


if __name__ == "__main__":
    main()
