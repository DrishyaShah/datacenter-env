"""
inference.py — DC-OpenEnv Baseline Inference Script V2
=======================================================
Runs all three datacenter cooling tasks against an LLM agent via OpenAI-compatible client.

Environment variables required:
    API_BASE_URL   e.g. https://api.groq.com/openai/v1
    MODEL_NAME     e.g. llama-3.3-70b-versatile
    HF_TOKEN       Your Groq (or other provider) API key

Usage:
    export HF_TOKEN='gsk_...'
    export API_BASE_URL='https://api.groq.com/openai/v1'
    export MODEL_NAME='llama-3.3-70b-versatile'
    python inference.py

Stdout format (strict — required by OpenEnv judges):
    [START] task=<task> env=dc-openenv model=<model>
    [STEP]  step=<n> action=<json> reward=<f> done=<bool> error=<str|null>
    [END]   success=<bool> steps=<n> score=<f> rewards=<csv>
"""

import os
import sys
import json
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from server.environment import DCEnvironment
from server.models import DCAction, DCObservation, ZoneAdjustment

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set.\n"
        "Export it before running:\n"
        "  export HF_TOKEN='gsk_...'          # bash/zsh\n"
        "  $env:HF_TOKEN='gsk_...'            # PowerShell"
    )

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SUCCESS_THRESHOLD = 0.6


# ── Task registry ─────────────────────────────────────────────────────────────
# All three tasks are live. No stubs.
TASKS = [
    {
        "name":        "easy-single-zone",
        "description": "Single-zone thermal runaway recovery under steady load",
        "max_steps":   48,
    },
    {
        "name":        "medium-multi-zone",
        "description": "3-zone load surge with faulty sensor and diurnal variation",
        "max_steps":   144,
    },
    {
        "name":        "hard-cascading-failure",
        "description": "4-zone cascading chiller failure with carbon-aware triage",
        "max_steps":   288,
    },
]


# ── System prompts ────────────────────────────────────────────────────────────
# Each task gets a tailored system prompt. Shared preamble is factored out.

_SHARED_PREAMBLE = textwrap.dedent("""
    You are a data centre operations engineer AI managing a physical cooling system.

    YOUR PRIMARY OBJECTIVES (in priority order):
      1. Keep ALL zone temperatures strictly between 18°C and 27°C.
      2. Minimise PUE (Power Usage Effectiveness) — lower is better; 1.0 is perfect.
      3. Minimise carbon emissions — reduce cooling power during high-carbon grid periods.
      4. Maintain smooth, stable control — avoid large abrupt changes every step.

    ── ACTION FORMAT ────────────────────────────────────────────────────────────
    You MUST respond with ONLY a valid JSON object. No markdown fences, no prose outside JSON.

    {
      "zone_adjustments": [
        {
          "zone_id": "<zone_id>",
          "fan_speed_pct": <number 0–100>,
          "supply_air_temp_setpoint_c": <number 16–26>
        }
      ],
      "chiller_setpoint_c": <number 6–15>,
      "chiller_active": <true|false>,
      "reasoning": "<one or two sentences explaining your decision>"
    }

    You MUST include every visible zone in zone_adjustments.

    ── ACTION FIELDS EXPLAINED ──────────────────────────────────────────────────
    fan_speed_pct [0–100]:
      Controls airflow volume. Higher = more heat removed per minute.
      Fan power scales as the CUBE of speed — running at 100% costs ~8× more than 50%.
      Avoid blasting fans at 100% indefinitely; dial back once the zone is stable.

    supply_air_temp_setpoint_c [16–26]:
      Controls how cold the delivered air is. Lower = more cooling power but higher chiller load.
      Typical efficient range: 18–22°C. Use lower values only when recovering from overheating.

    chiller_setpoint_c [6–15]:
      Controls chilled-water temperature for the whole facility.
      Lower = colder water = more cooling capacity, but lower chiller COP (less efficient).
      Efficient range: 10–13°C. Only go below 10°C during emergencies.

    chiller_active [true|false]:
      Turning OFF the chiller saves significant power but temperatures will rise for
      3–5 steps before you feel the consequence. Only turn off when free-cooling is available
      (free_cooling_potential > 0.5) or during low-load, cool-outdoor conditions.

    ── RATE LIMITS (enforced by the simulator) ───────────────────────────────────
    The simulator clips actions that change too fast:
      - fan_speed_pct:             max ±20% per step
      - supply_air_temp_setpoint_c: max ±2°C per step
      - chiller_setpoint_c:         max ±1°C per step
    Plan ahead. You cannot make large instantaneous corrections.

    ── SENSOR CONFIDENCE ────────────────────────────────────────────────────────
    Each zone reports sensor_confidence [0.0–1.0].
      1.0 = fully reliable reading.
      < 0.5 = sensor is drifting — DO NOT trust reported_temp_c for this zone.
    When sensor_confidence is low, infer the zone's true temperature from:
      - its IT load (higher load → more heat → higher true temp)
      - the hot_aisle_temp_c reading (less prone to drift than cold-aisle sensor)
      - physics: if you are supplying cold air and fans are high, the zone cannot be as
        hot as the faulty sensor claims.

    ── CARBON PENALTY INTERPRETATION ────────────────────────────────────────────
    carbon_intensity_normalized [0.0–1.0] measures grid dirtiness right now.
      low (< 0.25):          Use cooling freely — cheap, clean power available.
      medium (0.25–0.55):    Normal operation.
      high (0.55–0.80):      Prefer efficiency. Raise supply air temp slightly, avoid
                             unnecessary chiller load.
      critical_high (> 0.80): Aggressively reduce cooling power for non-critical zones.
                              Accept zone_storage / zone_infra running warm (up to 26°C).
                              Never compromise critical zones (zone_priority = 2).

    ── ZONE PRIORITY ─────────────────────────────────────────────────────────────
    zone_priority: 0 = low, 1 = medium, 2 = critical.
    Critical zones (priority 2) must NEVER exceed 30°C.
    Low-priority zones can be sacrificed (allowed to run up to 26°C) to protect critical ones
    when cooling capacity is limited.
""").strip()

_EASY_ADDENDUM = textwrap.dedent("""
    ── EASY TASK NOTES ──────────────────────────────────────────────────────────
    Single zone. Zone starts overheating (≈28.5°C). You have full chiller capacity.
    Strategy:
      1. Aggressively cool for the first 5–6 steps (high fan, low supply temp).
      2. Once temperature is in range (18–27°C), dial back fan speed to minimise PUE.
      3. Do not keep fans at 100% once stable — this wastes energy and hurts your score.
""").strip()

_MEDIUM_ADDENDUM = textwrap.dedent("""
    ── MEDIUM TASK NOTES ────────────────────────────────────────────────────────
    Three zones. One zone (zone_ai) has a FAULTY sensor — it will report temperatures
    9–12°C HIGHER than actual. Watch sensor_confidence: when it drops below 0.3, ignore
    reported_temp_c entirely for that zone and reason from load and hot_aisle_temp_c.

    IT load follows a diurnal curve — it will surge (60% → 95%) between steps 30–60.
    Watch load_curve_phase: when it says "ramp_up", PRE-COOL the facility BEFORE the
    surge arrives. Do not wait until temperatures are already rising — thermal inertia
    means you are always 2–3 steps behind if you react rather than anticipate.
""").strip()

_HARD_ADDENDUM = textwrap.dedent("""
    ── HARD TASK NOTES ──────────────────────────────────────────────────────────
    Four zones. Two are critical (zone_ai_1, zone_ai_2). At step ~15, the chiller will
    begin to fail (watch chiller_fault_detected = true and chiller_cop dropping).
    At step ~20, the chiller goes fully offline.

    AFTER CHILLER FAILURE:
      - You have only fan cooling + free-air cooling (check free_cooling_potential).
      - TRIAGE: reduce fan speed on low-priority zones (zone_infra, zone_storage) to
        maintain airflow to the critical zones. It is acceptable — even correct — to let
        zone_infra run warm to protect zone_ai_*.
      - If free_cooling_potential > 0.4 and outside_temp_c < 20°C, you can keep
        chiller_active = false and rely on economiser cooling.

    CARBON-AWARE TRIAGE:
      - During critical_high carbon windows (steps 80–160), avoid using backup cooling
        that draws dirty power for non-critical zones.
      - During low carbon windows (steps 0–40 and 200–288), use cooling freely.

    REASONING IS GRADED:
      Your "reasoning" field is evaluated. Say what you are actually doing and why.
      Inconsistency between stated reasoning and actual action is penalised.
      Example of good reasoning: "Chiller COP at 1.2 — pre-cooling critical zones before
      expected full failure at step 20. Reducing zone_infra fan to redirect capacity."
""").strip()

SYSTEM_PROMPTS = {
    "easy-single-zone":      f"{_SHARED_PREAMBLE}\n\n{_EASY_ADDENDUM}",
    "medium-multi-zone":     f"{_SHARED_PREAMBLE}\n\n{_MEDIUM_ADDENDUM}",
    "hard-cascading-failure": f"{_SHARED_PREAMBLE}\n\n{_HARD_ADDENDUM}",
}


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, model: str):
    print(f"[START] task={task} env=dc-openenv model={model}", flush=True)

def log_step(step: int, action_json: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_json} "
        f"reward={reward:.4f} done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_csv = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_csv}",
        flush=True,
    )


# ── Observation serialisation ─────────────────────────────────────────────────

def obs_to_dict(obs: DCObservation) -> Dict[str, Any]:
    """Serialise DCObservation → dict for LLM consumption (all V2 fields included)."""
    return {
        "step": obs.step,
        "timestamp_hour": obs.timestamp_hour,
        "outside_temp_c": obs.outside_temp_c,
        "wet_bulb_temp_c": obs.wet_bulb_temp_c,
        "current_pue": obs.current_pue,
        "free_cooling_potential": obs.free_cooling_potential,
        "chiller_active": obs.chiller_active,
        "chiller_setpoint_c": obs.chiller_setpoint_c,
        "chiller_cop": obs.chiller_cop,
        "chiller_fault_detected": obs.chiller_fault_detected,
        "grid_carbon_intensity": obs.grid_carbon_intensity,
        "carbon_intensity_normalized": obs.carbon_intensity_normalized,
        "load_curve_phase": obs.load_curve_phase,
        "sla_violation_streak": obs.sla_violation_streak,
        "maintenance_active": obs.maintenance_active,
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_priority": z.zone_priority,
                "reported_temp_c": z.reported_temp_c,
                "cold_aisle_temp_c": z.cold_aisle_temp_c,
                "hot_aisle_temp_c": z.hot_aisle_temp_c,
                "supply_air_temp_c": z.supply_air_temp_c,
                "supply_air_temp_setpoint_c": z.supply_air_temp_setpoint_c,
                "fan_speed_pct": z.fan_speed_pct,
                "it_load_kw": z.it_load_kw,
                "it_load_pct": z.it_load_pct,
                "load_forecast_next_hour": z.load_forecast_next_hour,
                "humidity_pct": z.humidity_pct,
                "sensor_confidence": z.sensor_confidence,
            }
            for z in obs.zones
        ],
        "history": obs.history[-2:],   # last 2 snapshots to keep prompt size manageable
        "maintenance_notes": obs.maintenance_notes,
        "upcoming_events": obs.upcoming_events,
    }


# ── LLM call ──────────────────────────────────────────────────────────────────

def get_llm_action(
    obs_dict: Dict[str, Any],
    step: int,
    history_summary: List[str],
    task: str,
) -> Dict[str, Any]:
    history_block = "\n".join(history_summary[-4:]) if history_summary else "None"
    user_prompt = textwrap.dedent(f"""
        Step {step} — Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent action history:
        {history_block}

        Respond now with your JSON action.
    """).strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[task]},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Action construction ───────────────────────────────────────────────────────

def build_action(llm_result: Dict[str, Any], obs: DCObservation) -> DCAction:
    """
    Convert LLM JSON output → validated DCAction.

    Falls back to safe defaults per zone if the LLM omits or garbles any field.
    """
    zone_ids = [z.zone_id for z in obs.zones]
    current_fans = {z.zone_id: z.fan_speed_pct for z in obs.zones}
    current_supply = {z.zone_id: z.supply_air_temp_setpoint_c for z in obs.zones}

    adjustments_raw = llm_result.get("zone_adjustments", [])
    seen_zones = set()
    adjustments = []

    for adj in adjustments_raw:
        zid = adj.get("zone_id", "")
        if zid not in zone_ids or zid in seen_zones:
            continue
        seen_zones.add(zid)
        try:
            adjustments.append(ZoneAdjustment(
                zone_id=zid,
                fan_speed_pct=float(adj.get("fan_speed_pct", current_fans[zid])),
                supply_air_temp_setpoint_c=float(
                    adj.get("supply_air_temp_setpoint_c", current_supply[zid])
                ),
            ))
        except Exception:
            # Keep current settings for this zone on parse failure
            adjustments.append(ZoneAdjustment(
                zone_id=zid,
                fan_speed_pct=current_fans[zid],
                supply_air_temp_setpoint_c=current_supply[zid],
            ))

    # Ensure every zone is covered (LLM may have omitted some)
    for zid in zone_ids:
        if zid not in seen_zones:
            adjustments.append(ZoneAdjustment(
                zone_id=zid,
                fan_speed_pct=current_fans[zid],
                supply_air_temp_setpoint_c=current_supply[zid],
            ))

    # Facility-level levers — fall back to current observed values
    chiller_setpoint = float(
        llm_result.get("chiller_setpoint_c", obs.chiller_setpoint_c)
    )
    chiller_active = bool(llm_result.get("chiller_active", obs.chiller_active))

    return DCAction(
        zone_adjustments=adjustments,
        chiller_setpoint_c=max(6.0, min(15.0, chiller_setpoint)),
        chiller_active=chiller_active,
        reasoning=llm_result.get("reasoning", ""),
    )


def _fallback_action(obs: DCObservation) -> DCAction:
    """Safe fallback action when the LLM call fails — hold current settings."""
    return DCAction(
        zone_adjustments=[
            ZoneAdjustment(
                zone_id=z.zone_id,
                fan_speed_pct=z.fan_speed_pct,
                supply_air_temp_setpoint_c=z.supply_air_temp_setpoint_c,
            )
            for z in obs.zones
        ],
        chiller_setpoint_c=obs.chiller_setpoint_c,
        chiller_active=obs.chiller_active,
        reasoning="fallback: LLM error — holding current settings",
    )


def _action_to_log_json(action: DCAction) -> str:
    """Compact JSON string for the [STEP] log line."""
    return json.dumps(
        {
            "zone_adjustments": [
                {
                    "zone_id": a.zone_id,
                    "fan_speed_pct": a.fan_speed_pct,
                    "supply_air_temp_setpoint_c": a.supply_air_temp_setpoint_c,
                }
                for a in action.zone_adjustments
            ],
            "chiller_setpoint_c": action.chiller_setpoint_c,
            "chiller_active": action.chiller_active,
        },
        separators=(",", ":"),
    )


# ── Single task episode ───────────────────────────────────────────────────────

def run_task(task_cfg: Dict[str, Any]) -> float:
    task_name = task_cfg["name"]
    max_steps  = task_cfg["max_steps"]

    log_start(task=task_name, model=MODEL_NAME)

    env             = DCEnvironment(task=task_name)
    rewards: List[float] = []
    history_summary: List[str] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        reset_result = env.reset()
        obs: DCObservation = reset_result.observation

        for step in range(1, max_steps + 1):
            obs_dict  = obs_to_dict(obs)
            error_str: Optional[str] = None

            # LLM decision
            try:
                llm_result = get_llm_action(obs_dict, step, history_summary, task_name)
                action     = build_action(llm_result, obs)
            except Exception as exc:
                error_str = str(exc)[:120]
                action    = _fallback_action(obs)

            action_json = _action_to_log_json(action)

            # Step environment
            try:
                step_result = env.step(action)
                reward      = step_result.reward
                done        = step_result.done
                info        = step_result.info
                obs         = step_result.observation
            except Exception as exc:
                reward    = 0.0
                done      = True
                info      = {}
                error_str = (error_str or "") + f" | env_step: {str(exc)[:80]}"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action_json=action_json, reward=reward,
                     done=done, error=error_str)

            # Compact history line for next prompt
            zone_summary = ", ".join(
                f"{z.zone_id}(p{z.zone_priority})="
                f"{z.cold_aisle_temp_c:.1f}°C "
                f"conf={z.sensor_confidence:.2f} "
                f"fan={z.fan_speed_pct:.0f}%"
                for z in obs.zones
            )
            history_summary.append(
                f"Step {step}: {zone_summary} | "
                f"pue={obs.current_pue:.3f} | "
                f"carbon={obs.grid_carbon_intensity} | "
                f"reward={reward:.4f}"
            )

            if done:
                score = info.get("final_score", _normalise_rewards(rewards))
                break

        if not rewards:
            score = 0.0
        elif score == 0.0:
            # Episode hit max_steps without early termination — get final score from grader
            final_info = getattr(env, "_grader", None)
            if final_info is not None:
                try:
                    score = env._grader.final_score()
                except Exception:
                    score = _normalise_rewards(rewards)
            else:
                score = _normalise_rewards(rewards)

        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Fatal error in task {task_name}: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def _normalise_rewards(rewards: List[float]) -> float:
    """Fallback score from raw reward list when grader is unavailable."""
    if not rewards:
        return 0.0
    return round(max(0.0, min(1.0, (sum(rewards) / len(rewards) + 1.0) / 2.0)), 4)


# ── Entry point ───────────────────────────────────────────────────────────────
def ask_to_continue(task_name: str) -> bool:
    """Prompt user whether to run a task."""
    while True:
        ans = input(f"Do you want to run '{task_name}'? (yes/no): ").strip().lower()
        if ans in ("yes", "y"):
            return True
        elif ans in ("no", "n"):
            return False
        else:
            print("Please answer 'yes' or 'no'.")



def main():
    print(f"[INFO] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}",   flush=True)
    print(f"[INFO] Running {len(TASKS)} tasks\n",  flush=True)

    all_scores: List[float] = []

    for task_cfg in TASKS:
        task_name = task_cfg["name"]

        # Always run easy
        if task_name == "easy-single-zone":
            run = True
        else:
            # Ask user for medium & hard
            run = ask_to_continue(task_name)

        if not run:
            print(f"[SKIP] {task_name}\n", flush=True)
            continue

        score = run_task(task_cfg)
        all_scores.append(score)
        print(f"[SCORE] {task_name} => {score:.4f}\n", flush=True)

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"[FINAL] overall_score={overall:.4f}", flush=True)
    else:
        print("[FINAL] No tasks were executed.", flush=True)

if __name__ == "__main__":
    main()