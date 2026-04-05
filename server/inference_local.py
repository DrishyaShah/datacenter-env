"""
inference_local.py — DC-OpenEnv Local Inference Script with Final Scoring
========================================================================
Runs the datacenter environment fully locally with Gemini LLM agent.
Logs [START], [STEP], [END] per OpenEnv spec.

Requirements:
  GEMINI_API_KEY   Your Gemini API key
  MODEL_NAME       Model identifier (default: gemini-2.0-flash)
"""

import os
import json
import textwrap
import time
from typing import List, Optional
import requests

from datacenter_env.server.environment import DCEnvironment
from datacenter_env.server.models import DCAction, DCObservation

# ── Config ────────────────────────────────────────────────────────────────

GEMINI_API_KEY = "YOUR API KEY HERE"
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

MAX_STEPS = 12
SUCCESS_SCORE_THRESHOLD = 0.6
TEMPERATURE = 0.3
MAX_TOKENS = 512

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data centre operations engineer AI assistant.
    You manage cooling systems in server rooms to:
    1. Keep all zone temperatures between 18C and 27C
    2. Minimize PUE (Power Usage Effectiveness)
    3. Consider maintenance notes and upcoming events

    Respond ONLY with a JSON object:
    {
      "zone_adjustments": [{"zone_id": "zone_1", "fan_speed_pct": <0-100>}],
      "reasoning": "Brief explanation"
    }
""").strip()

# ── Helpers ────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def observation_to_dict(obs: DCObservation) -> dict:
    return {
        "zones": [zone if isinstance(zone, dict) else zone.__dict__ for zone in getattr(obs, "zones", [])],
        "reward": getattr(obs, "reward", 0.0),
        "done": getattr(obs, "done", False),
        "metadata": getattr(obs, "metadata", {}),
    }

# ── Gemini API ─────────────────────────────────────────────────────────────

def query_gemini(prompt: str) -> dict:
    url = f"{GEMINI_API_BASE}/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
        }
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()

        data = r.json()
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        return json.loads(response_text.strip())

    except Exception as e:
        print(f"[DEBUG] Gemini parse error: {e}", flush=True)
        return {}

# ── Agent ──────────────────────────────────────────────────────────────────

def get_agent_action(observation: DCObservation, step: int, history: List[str]) -> DCAction:
    obs_dict = observation_to_dict(observation)
    history_block = "\n".join(history[-3:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Step {step} - Current Data Centre State:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Respond with your JSON action now.
    """).strip()

    result = query_gemini(user_prompt)

    zones = obs_dict.get("zones", [])
    zone_adjustments = result.get(
        "zone_adjustments",
        [{"zone_id": z["zone_id"], "fan_speed_pct": 80.0} for z in zones]
    )

    reasoning = result.get("reasoning", "fallback: increase cooling")

    return DCAction(zone_adjustments=zone_adjustments, reasoning=reasoning)

# ── Main Loop ──────────────────────────────────────────────────────────────

def main():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    env = DCEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task="easy-single-zone", env="dc-openenv-local", model=MODEL_NAME)

    try:
        obs: DCObservation = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action = get_agent_action(obs, step, history)
            action_str = json.dumps({"adjustments": action.zone_adjustments}).replace(" ", "")

            try:
                step_result: DCObservation = env.step(action)
                reward = getattr(step_result, "reward", 0.0)
                done = getattr(step_result, "done", False)
                obs = step_result
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)[:100]
            else:
                error = None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            zones_summary = ", ".join(
                f"{z['zone_id']}={z.get('temp_c','?')}C fan={z.get('fan_speed_pct','?')}%"
                for z in observation_to_dict(obs)["zones"]
            )
            history.append(f"Step {step}: {zones_summary} reward={reward:.2f}")

            # ✅ RATE LIMIT FIX
            time.sleep(4)

            if done:
                break

        if hasattr(env, "_grader_state") and env._grader_state is not None:
            grader = env._grader_state
            score = grader.steps_in_range / max(1, grader.steps_total)
        else:
            score = max(0.0, min(1.0, (sum(rewards)/len(rewards)+1)/2.0))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()