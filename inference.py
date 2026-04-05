"""
inference.py — DC-OpenEnv Baseline Inference Script
====================================================
Runs the datacenter cooling environment against an LLM agent via OpenAI-compatible client.

Environment variables required:
    API_BASE_URL   e.g. https://api.groq.com/openai/v1
    MODEL_NAME     e.g. llama-3.3-70b-versatile
    HF_TOKEN       Your Groq (or other provider) API key

Usage:
    $env:HF_TOKEN="gsk_..."
    $env:API_BASE_URL="https://api.groq.com/openai/v1"
    $env:MODEL_NAME="llama-3.3-70b-versatile"
    python inference.py

Stdout format (strict — do not alter):
    [START] task=<task> env=dc-openenv model=<model>
    [STEP]  step=<n> action=<json> reward=<f> done=<bool> error=<str|null>
    [END]   success=<bool> steps=<n> score=<f> rewards=<csv>
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Path setup: inference.py lives in root, env code lives in server/ ─────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from server.datacenter_env_environment import DCEnvironment
from server.models import DCAction, DCObservation, ZoneAdjustment
from server.grader_easy import compute_final_score

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set.\n"
        "  PowerShell: $env:HF_TOKEN='gsk_...'\n"
        "  bash/zsh:   export HF_TOKEN='gsk_...'"
    )

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SUCCESS_THRESHOLD = 0.6

# ── Task registry ──────────────────────────────────────────────────────────────
TASKS = [
    {
        "name":        "easy-single-zone",
        "description": "Single-zone thermal runaway recovery under steady load",
        "max_steps":   48,
        "stub":        False,
    },
    {
        "name":        "medium-multi-zone",
        "description": "Multi-zone load surge with faulty sensor and diurnal variation",
        "max_steps":   20,
        "stub":        True,
    },
    {
        "name":        "hard-cascading-failure",
        "description": "Cascading chiller failure with carbon-aware triage",
        "max_steps":   30,
        "stub":        True,
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────
# NOTE: Zone IDs are NOT hardcoded here.
# The LLM reads actual zone_id values from the observation JSON.
# This prevents zone_id mismatch between prompt defaults and scenario zone names.
SYSTEM_PROMPT = textwrap.dedent("""
    You are a data centre operations engineer AI managing server room cooling.

    Your goals (in priority order):
      1. Keep ALL zone temperatures strictly between 18C and 27C (critical)
      2. Minimise PUE (Power Usage Effectiveness) — lower is better
      3. Avoid humidity violations (keep between 40% and 60%)

    At each step you receive the current data centre state as JSON.
    The "zones" array lists each zone with its zone_id and sensor readings.

    You MUST respond with ONLY a valid JSON object — no markdown, no text outside JSON:
    {
      "zone_adjustments": [
        {
          "zone_id": "<exact zone_id copied from the observation>",
          "fan_speed_pct": <number 0.0 to 100.0>
        }
      ],
      "reasoning": "<one sentence explaining your decision>"
    }

    IMPORTANT RULES:
    - Copy the EXACT zone_id string from the observation — do not invent or rename zones
    - Include ALL zones from the observation in your zone_adjustments every step
    - If a zone reported_temp_c is ABOVE 27C: set fan_speed_pct high (80 to 100)
    - If a zone reported_temp_c is BELOW 18C: set fan_speed_pct low (20 to 40)
    - If temperature is in safe range (18-27C): use moderate fan speed (40-65) to save energy
    - The zone starts OVERHEATING above 27C — act aggressively in early steps to cool it down
""").strip()


# ── Logging helpers ────────────────────────────────────────────────────────────
# FIX 3: _logged_steps prevents any duplicate step lines per episode
_logged_steps: set = set()


def log_start(task: str, model: str):
    global _logged_steps
    _logged_steps = set()
    print(f"[START] task={task} env=dc-openenv model={model}", flush=True)


def log_step(step: int, action_json: str, reward: float, done: bool, error: Optional[str]):
    if step in _logged_steps:
        return  # deduplicate — never emit same step twice
    _logged_steps.add(step)
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


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_llm_action(obs_dict: dict, step: int, history: List[str]) -> dict:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Step {step} — Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Respond with your JSON action now. Remember to use the exact zone_id from the state above.
    """).strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model wrapped response
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
    """
    FIX 1 — Zone ID remapping:
    If the LLM sends a zone_id not present in the environment,
    remap it positionally to the correct zone_id.
    Prevents actions being silently dropped due to LLM hallucinating zone names.
    """
    adjustments_raw = llm_result.get("zone_adjustments", [])
    adjustments = []

    for i, adj in enumerate(adjustments_raw):
        llm_zone_id = adj.get("zone_id", "")
        fan_speed   = float(adj.get("fan_speed_pct", 80.0))

        if llm_zone_id in zone_ids:
            actual_zone_id = llm_zone_id          # LLM got it right
        elif i < len(zone_ids):
            actual_zone_id = zone_ids[i]          # positional remap
        else:
            continue                               # invented extra zone, skip

        adjustments.append(ZoneAdjustment(
            zone_id=actual_zone_id,
            fan_speed_pct=max(0.0, min(100.0, fan_speed)),
        ))

    # Ensure every real zone is covered even if LLM omitted some
    covered = {a.zone_id for a in adjustments}
    for zone_id in zone_ids:
        if zone_id not in covered:
            adjustments.append(ZoneAdjustment(zone_id=zone_id, fan_speed_pct=85.0))

    return DCAction(
        zone_adjustments=adjustments,
        reasoning=llm_result.get("reasoning", ""),
    )


# ── Run one task episode (strictly sequential — no threads) ───────────────────
def run_task(task_cfg: dict) -> float:
    task_name = task_cfg["name"]
    max_steps  = task_cfg["max_steps"]

    log_start(task=task_name, model=MODEL_NAME)

    # STUB tasks: emit one step and return
    if task_cfg["stub"]:
        log_step(
            step=1,
            action_json="{}",
            reward=0.0,
            done=True,
            error=f"STUB: {task_cfg['description']} not yet implemented",
        )
        log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        return 0.0

    # LIVE task
    env         = DCEnvironment(task=task_name)
    rewards:    List[float] = []
    history:    List[str]   = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        reset_result       = env.reset()
        obs: DCObservation = reset_result.observation

        for step in range(1, max_steps + 1):

            # Build observation dict for LLM
            obs_dict = {
                "step":                  obs.step,
                "timestamp_hour":        obs.timestamp_hour,
                "outside_temp_c":        obs.outside_temp_c,
                "current_pue":           obs.current_pue,
                "chiller_active":        obs.chiller_active,
                "chiller_cop":           obs.chiller_cop,
                "grid_carbon_intensity": obs.grid_carbon_intensity,
                "zones": [
                    {
                        "zone_id":         z.zone_id,         # real zone_id from env
                        "reported_temp_c": z.reported_temp_c,
                        "fan_speed_pct":   z.fan_speed_pct,
                        "it_load_kw":      z.it_load_kw,
                        "humidity_pct":    z.humidity_pct,
                        "setpoint_c":      z.setpoint_c,
                    }
                    for z in obs.zones
                ],
                "maintenance_notes": obs.maintenance_notes,
                "upcoming_events":   obs.upcoming_events,
            }
            zone_ids = [z.zone_id for z in obs.zones]

            # Get LLM action
            error_str = None
            try:
                llm_result = get_llm_action(obs_dict, step, history)
                action     = build_action(llm_result, zone_ids)
            except Exception as e:
                error_str = str(e)[:120]
                action = DCAction(
                    zone_adjustments=[
                        ZoneAdjustment(zone_id=z, fan_speed_pct=85.0)
                        for z in zone_ids
                    ],
                    reasoning="fallback: LLM parse error",
                )

            action_json = json.dumps(
                {"zone_adjustments": [a.model_dump() for a in action.zone_adjustments]},
                separators=(",", ":"),
            )

            # Step the environment
            done = False
            try:
                step_result = env.step(action)
                reward      = step_result.reward
                done        = step_result.done
                obs         = step_result.observation
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

            zone_summary = ", ".join(
                f"{z.zone_id}={z.reported_temp_c:.1f}C fan={z.fan_speed_pct:.0f}%"
                for z in obs.zones
            )
            history.append(
                f"Step {step}: {zone_summary} | reward={reward:.4f} | pue={obs.current_pue:.3f}"
            )

            if done:
                break

        # Compute final score via grader
        grader = getattr(env, "_grader_state", None)
        if grader and getattr(grader, "steps_total", 0) > 0:
            score = compute_final_score(grader)
        elif rewards:
            score = max(0.0, min(1.0, (sum(rewards) / len(rewards) + 1.0) / 2.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error in task {task_name}: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}",   flush=True)
    print(f"[INFO] Running {len(TASKS)} tasks\n",  flush=True)

    all_scores = []

    # Strictly sequential — one task completes fully before the next starts
    for task_cfg in TASKS:
        score = run_task(task_cfg)
        all_scores.append(score)
        print(f"[SCORE] {task_cfg['name']} => {score:.4f}\n", flush=True)

    overall = sum(all_scores) / len(all_scores)
    print(f"[FINAL] overall_score={overall:.4f}", flush=True)


if __name__ == "__main__":
    main()