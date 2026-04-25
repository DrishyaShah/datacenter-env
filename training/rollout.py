"""
GRPO rollout collection for ClusterEnv scheduler training.

Public API:
    parse_decisions(completion, window_state)  -> (list[AdmissionDecision], float)
    compute_window_reward(...)                 -> float
    collect_rollouts(generate_fn, ...)         -> list[dict]
    compute_grpo_advantages(rollouts)          -> list[float]
"""

from __future__ import annotations

import json
import re
import os
import sys
from collections import defaultdict
from typing import Callable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.cluster_environment import ClusterEnvironment
from server.economic import WindowState
from server.economic.job_request import AdmissionDecision
from server.agents.baseline_scheduler import priority_weighted_threshold
from training.prompts import build_prompt


# -- JSON extraction -----------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Extract first {...} block from text and parse it."""
    m = _JSON_RE.search(text)
    if m is None:
        raise ValueError("no JSON block found")
    return json.loads(m.group())


# -- Parser --------------------------------------------------------------------


def parse_decisions(
    completion: str,
    window_state: WindowState,
) -> tuple[list[AdmissionDecision], float]:
    """
    Parse LLM completion into AdmissionDecision list.

    Expected completion format:
        {"decisions": [...], "operator_note": "..."}

    Returns (decisions, parse_penalty) where parse_penalty is 0.0 on success
    or -0.5 on any parse failure (fallback: priority_weighted_threshold).
    """
    valid_ids = {r.request_id for r in window_state.all_pending}

    try:
        data    = _extract_json(completion)
        entries = data["decisions"]
        decisions: list[AdmissionDecision] = []

        for e in entries:
            rid = e.get("request_id", "")
            if rid not in valid_ids:
                continue  # hallucinated or stale id -- skip silently
            try:
                dec = AdmissionDecision(
                    request_id       = rid,
                    decision         = str(e.get("decision", "")).upper(),
                    scheduled_window = e.get("scheduled_window"),
                    reasoning        = str(e.get("reasoning", "")),
                )
                decisions.append(dec)
            except (ValueError, KeyError):
                continue  # individual bad entry -- skip, don't abort

        return decisions, 0.0

    except Exception:
        return priority_weighted_threshold(window_state), -0.5


# -- Reward shaping ------------------------------------------------------------


def compute_window_reward(
    env_reward: float,
    ws_before: WindowState,
    decisions: list[AdmissionDecision],
    parse_penalty: float,
) -> float:
    """
    env_reward  - 0.05 * n_accepted_flagged_jobs + parse_penalty

    Penalises the LLM for accepting jobs from teams with active "flag" or
    "escalate" oversight flags (gaming behaviour the scheduler should resist).
    """
    flagged_teams = {
        f.team_id
        for f in ws_before.oversight_flags
        if f.severity in ("flag", "escalate")
    }
    if flagged_teams:
        req_index = {r.request_id: r for r in ws_before.all_pending}
        n_accepted_flagged = sum(
            1
            for d in decisions
            if d.decision == "ACCEPT"
            and req_index.get(d.request_id) is not None
            and req_index[d.request_id].team_id in flagged_teams
        )
    else:
        n_accepted_flagged = 0

    return env_reward - 0.05 * n_accepted_flagged + parse_penalty


# -- Episode runner ------------------------------------------------------------


def collect_rollouts(
    generate_fn: Callable[[str], str],
    n_episodes: int = 4,
    base_seed: int = 0,
    enable_chiller_fault: bool = True,
) -> list[dict]:
    """
    Run n_episodes complete episodes (8 windows each) and collect
    (prompt, completion, reward, window_idx, episode_idx) records.

    Returns list of length n_episodes * 8.

    PPOCoolingController is loaded once and shared across all episodes
    to save memory and startup time.
    """
    from server.agents.ppo_cooling_controller import PPOCoolingController

    ctrl = PPOCoolingController()
    rollouts: list[dict] = []

    for ep in range(n_episodes):
        env  = ClusterEnvironment(
            cooling_controller  = ctrl,
            enable_chiller_fault = enable_chiller_fault,
        )
        ws   = env.reset(seed=base_seed + ep)
        done = False

        while not done:
            prompt     = build_prompt(ws)
            completion = generate_fn(prompt)

            decisions, parse_penalty = parse_decisions(completion, ws)

            ws_before           = ws
            ws, env_reward, done, _ = env.step(decisions)

            reward = compute_window_reward(
                env_reward, ws_before, decisions, parse_penalty
            )
            rollouts.append({
                "prompt":      prompt,
                "completion":  completion,
                "reward":      reward,
                "window_idx":  ws_before.window_idx,
                "episode_idx": ep,
            })

    return rollouts


# -- GRPO advantage computation ------------------------------------------------


def compute_grpo_advantages(rollouts: list[dict]) -> list[float]:
    """
    Normalise rewards within each window-index group (GRPO-style).

    Group = all rollouts with the same window_idx (one per episode per window).
    Advantage = (reward - group_mean) / (group_std + eps).

    Returns a list of floats in the same order as rollouts.
    """
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, r in enumerate(rollouts):
        groups[r["window_idx"]].append((i, r["reward"]))

    advantages = [0.0] * len(rollouts)
    for items in groups.values():
        idxs, rewards = zip(*items)
        arr  = np.array(rewards, dtype=np.float32)
        mean = arr.mean()
        std  = arr.std()
        normed = (arr - mean) / (std + 1e-8)
        for idx, adv in zip(idxs, normed):
            advantages[idx] = float(adv)

    return advantages
