"""
inference.py — DC-OpenEnv Baseline Inference Script
====================================================
Runs the datacenter cooling environment against an LLM agent via OpenAI-compatible client.

Environment variables required:
    API_BASE_URL   e.g. https://api.groq.com/openai/v1
    MODEL_NAME     e.g. llama-3.3-70b-versatile
    HF_TOKEN       Your Groq (or other provider) API key

Usage:
    # Set env vars first, then:
    python inference.py

Stdout format (strict):
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

# ── Path setup (inference.py lives in root, env code lives in server/) ────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from datacenter_env_environment import DCEnvironment
from models import DCAction, DCObservation, ZoneAdjustment

# ── Config (read from environment variables) ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set.\n"
        "Export it before running:\n"
        "  $env:HF_TOKEN='gsk_...'          # PowerShell\n"
        "  export HF_TOKEN='gsk_...'         # bash/zsh"
    )

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Task registry ─────────────────────────────────────────────────────────────
TASKS = [
    {
        "name":        "easy-single-zone",
        "description": "Single-zone temperature control under steady load",
        "max_steps":   12,
        "stub":        False,
    },
    {
        "name":        "medium-multi-zone",
        "description": "Multi-zone balancing with variable IT load (STUB)",
        "max_steps":   20,
        "stub":        True,
    },
    {
        "name":        "hard-cascading-failure",
        "description": "Cascading chiller failure with faulty sensors (STUB)",
        "max_steps":   30,
        "stub":        True,
    },
]

SUCCESS_THRESHOLD = 0.6

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a data centre operations engineer AI.
    Your job is to manage cooling systems to:
      1. Keep ALL zone temperatures strictly between 18°C and 27°C
      2. Minimise PUE (Power Usage Effectiveness) — lower is better
      3. Avoid humidity violations (keep between 40% and 60%)

    At each step you receive the current data centre state as JSON.
    You MUST respond with ONLY a valid JSON object — no markdown, no explanation outside JSON:
    {
      "zone_adjustments": [
        {"zone_id": "zone_1", "fan_speed_pct": <number 0-100>}
      ],
      "reasoning": "<one sentence>"
    }

    Rules:
    - If a zone is above 27°C → increase fan_speed_pct toward 100
    - If a zone is below 18°C → decrease fan_speed_pct
    - If temperature is in range → tune fan speed to minimise PUE
    - Always include every visible zone in zone_adjustments
""").strip()

# ── Logging helpers (strict format required by OpenEnv judges) ────────────────
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

# ── LLM call ─────────────────────────────────────────────────────────────────
def get_llm_action(obs_dict: dict, step: int, history: List[str]) -> dict:
    history_block = "\n".join(history[-4:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        Step {step} — Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Respond now with your JSON action.
    """).strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

# ── Build DCAction from LLM output ───────────────────────────────────────────
def build_action(llm_result: dict, zone_ids: List[str]) -> DCAction:
    adjustments_raw = llm_result.get("zone_adjustments", [])
    adjustments = []
    for adj in adjustments_raw:
        try:
            adjustments.append(ZoneAdjustment(
                zone_id=adj["zone_id"],
                fan_speed_pct=float(adj.get("fan_speed_pct", 80.0)),
            ))
        except Exception:
            continue

    # Fallback: if LLM returned nothing, push all zones to 85%
    if not adjustments:
        adjustments = [ZoneAdjustment(zone_id=z, fan_speed_pct=85.0) for z in zone_ids]

    return DCAction(
        zone_adjustments=adjustments,
        reasoning=llm_result.get("reasoning", ""),
    )

# ── Run a single task episode ─────────────────────────────────────────────────
def run_task(task_cfg: dict) -> float:
    task_name = task_cfg["name"]
    max_steps  = task_cfg["max_steps"]

    log_start(task=task_name, model=MODEL_NAME)

    # ── STUB tasks: log immediately and return 0.0 ────────────────────────────
    if task_cfg["stub"]:
        log_step(step=0, action="{}", reward=0.0, done=True,
                 error=f"STUB: {task_cfg['description']} not yet implemented")
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    # ── LIVE task ─────────────────────────────────────────────────────────────
    env     = DCEnvironment(task=task_name)
    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    score       = 0.0
    success     = False

    try:
        reset_result = env.reset()
        obs: DCObservation = reset_result.observation

        for step in range(1, max_steps + 1):
            # Build obs dict for LLM
            obs_dict = {
                "step":                obs.step,
                "timestamp_hour":      obs.timestamp_hour,
                "outside_temp_c":      obs.outside_temp_c,
                "current_pue":         obs.current_pue,
                "chiller_active":      obs.chiller_active,
                "grid_carbon_intensity": obs.grid_carbon_intensity,
                "zones": [
                    {
                        "zone_id":          z.zone_id,
                        "reported_temp_c":  z.reported_temp_c,
                        "fan_speed_pct":    z.fan_speed_pct,
                        "it_load_kw":       z.it_load_kw,
                        "humidity_pct":     z.humidity_pct,
                        "setpoint_c":       z.setpoint_c,
                    }
                    for z in obs.zones
                ],
                "maintenance_notes": obs.maintenance_notes,
                "upcoming_events":   obs.upcoming_events,
            }
            zone_ids = [z.zone_id for z in obs.zones]

            # LLM decision
            error_str = None
            try:
                llm_result = get_llm_action(obs_dict, step, history)
                action     = build_action(llm_result, zone_ids)
            except Exception as e:
                error_str  = str(e)[:120]
                action     = DCAction(
                    zone_adjustments=[ZoneAdjustment(zone_id=z, fan_speed_pct=85.0) for z in zone_ids],
                    reasoning="fallback: LLM error",
                )

            action_json = json.dumps({
                "zone_adjustments": [a.model_dump() for a in action.zone_adjustments]
            }, separators=(",", ":"))

            # Step environment
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
            log_step(step=step, action_json=action_json, reward=reward,
                     done=done, error=error_str)

            # Build history summary
            zone_summary = ", ".join(
                f"{z.zone_id}={z.reported_temp_c:.1f}°C fan={z.fan_speed_pct:.0f}%"
                for z in obs.zones
            )
            history.append(f"Step {step}: {zone_summary} | reward={reward:.4f} | pue={obs.current_pue:.3f}")

            if done:
                break

        # Compute final score
        grader = getattr(env, "_grader_state", None)
        if grader and grader.steps_total > 0:
            from grader_easy import compute_final_score
            score = compute_final_score(grader)
        elif rewards:
            score = max(0.0, min(1.0, (sum(rewards) / len(rewards) + 1.0) / 2.0))

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error in task {task_name}: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(f"[INFO] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}",   flush=True)
    print(f"[INFO] Running {len(TASKS)} tasks\n",  flush=True)

    all_scores = []
    for task_cfg in TASKS:
        score = run_task(task_cfg)
        all_scores.append(score)
        print(f"[SCORE] {task_cfg['name']} => {score:.4f}\n", flush=True)

    overall = sum(all_scores) / len(all_scores)
    print(f"[FINAL] overall_score={overall:.4f}", flush=True)

if __name__ == "__main__":
    main()