"""
inference.py — DC-OpenEnv Baseline Inference Script
====================================================
Runs the datacenter cooling environment against an LLM via an OpenAI-compatible API.

Environment variables:
    HF_TOKEN or OPENAI_API_KEY   API key (either is accepted)
    API_BASE_URL                 e.g. https://api.groq.com/openai/v1
    MODEL_NAME                   e.g. llama-3.3-70b-versatile
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
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.environment import DCEnvironment
from server.models import DCAction, DCObservation, ZoneAdjustment

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
VERBOSE = os.getenv("VERBOSE", "").strip().lower() in ("1", "true", "yes")

_steps_cap_raw = os.getenv("INFERENCE_MAX_STEPS_PER_TASK", "").strip()
INFERENCE_MAX_STEPS_PER_TASK: Optional[int] = (
    int(_steps_cap_raw) if _steps_cap_raw.isdigit() else None
)

if not API_KEY:
    raise RuntimeError(
        "Set HF_TOKEN or OPENAI_API_KEY.\n"
        "  PowerShell: $env:HF_TOKEN='...'\n"
        "  bash:       export HF_TOKEN='...'"
    )

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SUCCESS_THRESHOLD = 0.6

# ── Task registry (all three tasks run against DCEnvironment + graders) ───────
TASKS = [
    {
        "name": "easy-single-zone",
        "description": "Single-zone thermal runaway recovery under steady load",
        "max_steps": 48,
    },
    {
        "name": "medium-multi-zone",
        "description": "Multi-zone load surge with faulty sensor and diurnal variation",
        "max_steps": 144,
    },
    {
        "name": "hard-cascading-failure",
        "description": "Cascading chiller failure with carbon-aware triage",
        "max_steps": 288,
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a data centre operations engineer AI managing server room cooling.

    Your goals (in priority order):
      1. Keep ALL zone temperatures strictly between 18C and 27C (critical)
      2. Minimise PUE (Power Usage Effectiveness) — lower is better
      3. Keep humidity between 40% and 60%

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
      "reasoning": "<one sentence>"
    }

    RULES:
    - Copy EXACT zone_id strings from the observation — never invent zone names
    - Include ALL zones from the observation in zone_adjustments every step
    - Zone cold_aisle_temp_c ABOVE 27C → fan_speed_pct high (80-100), supply_air_temp_setpoint_c low (16-18)
    - Zone cold_aisle_temp_c BELOW 18C → fan_speed_pct low (20-40), supply_air_temp_setpoint_c high (22-26)
    - Zone in safe range → moderate fan (40-65) and supply_air_temp_setpoint_c near 20-22 to save energy
    - Keep chiller_active true unless you have a specific reason to disable it
    - Lower chiller_setpoint_c (e.g. 8.0) = colder water = more cooling power but higher energy cost
    - On overheating scenarios, act aggressively in early steps
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


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_llm_action(obs_dict: dict, step: int, history: List[str]) -> dict:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Step {step} — Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Respond with your JSON action. Use the exact zone_id values shown above.
    """).strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=350,
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


# ── Build DCAction from LLM output ────────────────────────────────────────────
def build_action(llm_result: dict, zone_ids: List[str]) -> DCAction:
    adjustments_raw = llm_result.get("zone_adjustments", [])
    adjustments = []

    for i, adj in enumerate(adjustments_raw):
        llm_zone_id = adj.get("zone_id", "")
        fan_speed = float(adj.get("fan_speed_pct", 80.0))
        supply_temp = float(adj.get("supply_air_temp_setpoint_c", 20.0))

        if llm_zone_id in zone_ids:
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
            adjustments.append(
                ZoneAdjustment(
                    zone_id=zone_id,
                    fan_speed_pct=85.0,
                    supply_air_temp_setpoint_c=18.0,
                )
            )

    return DCAction(
        zone_adjustments=adjustments,
        chiller_setpoint_c=float(llm_result.get("chiller_setpoint_c", 10.0)),
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
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_result = env.reset()
        obs: DCObservation = reset_result.observation

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
            zone_ids = [z.zone_id for z in obs.zones]

            error_str = None
            try:
                llm_result = get_llm_action(obs_dict, step, history)
                action = build_action(llm_result, zone_ids)
            except Exception as e:
                error_str = str(e)[:120]
                action = DCAction(
                    zone_adjustments=[
                        ZoneAdjustment(
                            zone_id=z,
                            fan_speed_pct=85.0,
                            supply_air_temp_setpoint_c=18.0,
                        )
                        for z in zone_ids
                    ],
                    chiller_setpoint_c=10.0,
                    chiller_active=True,
                    reasoning="fallback: LLM parse error",
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
                step_result = env.step(action)
                reward = step_result.reward
                done = step_result.done
                obs = step_result.observation
            except Exception as e:
                reward = 0.0
                done = True
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

            zone_summary = ", ".join(
                f"{z.zone_id}={z.cold_aisle_temp_c:.1f}C fan={z.fan_speed_pct:.0f}%"
                for z in obs.zones
            )
            history.append(
                f"Step {step}: {zone_summary} | reward={reward:.2f} | pue={obs.current_pue:.3f}"
            )

            if done:
                break

        grader = getattr(env, "_grader", None)
        if grader is not None and hasattr(grader, "final_score"):
            score = float(grader.final_score())
        elif rewards:
            score = max(0.0, min(1.0, (sum(rewards) / len(rewards) + 1.0) / 2.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        _vprint(f"[DEBUG] Fatal error in task {task_name}: {e}")

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    _vprint(f"[INFO] API_BASE_URL = {API_BASE_URL}")
    _vprint(f"[INFO] MODEL_NAME   = {MODEL_NAME}")
    _vprint(f"[INFO] Running {len(TASKS)} tasks")

    all_scores = []
    for task_cfg in TASKS:
        score = run_task(task_cfg)
        all_scores.append(score)
        _vprint(f"[SCORE] {task_cfg['name']} => {score:.2f}")

    overall = sum(all_scores) / len(all_scores)
    _vprint(f"[FINAL] overall_score={overall:.2f}")


if __name__ == "__main__":
    main()
