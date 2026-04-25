"""
ClusterEnv integration gate.

Run:
    python -m pytest tests/test_cluster.py -v
    python tests/test_cluster.py          # standalone with calibration output
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold, accept_all, reject_all


# ── Unit tests ────────────────────────────────────────────────────────────────


def test_env_reset_returns_window_state():
    """reset() returns a WindowState with window_idx=0 and pending jobs."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=0)
    assert obs.window_idx == 0
    assert obs.total_windows == 8
    assert len(obs.pending_requests) > 0, "window 0 should have pending jobs"
    assert obs.capacity_headroom_kw > 0


def test_full_episode_completes():
    """8-window episode runs to completion with no exceptions."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=42)
    done = False
    step_count = 0
    while not done:
        decisions = priority_weighted_threshold(obs)
        obs, reward, done, info = env.step(decisions)
        step_count += 1
        assert isinstance(reward, float)
    assert step_count == 8


def test_accept_all_causes_power_violations():
    """accept_all should trigger power budget violations (total load > 900 kW)."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=42)
    done = False
    while not done:
        decisions = accept_all(obs)
        obs, reward, done, info = env.step(decisions)

    ir = env._grader.incident_rate()
    assert ir > 0.3, (
        f"accept_all incident rate {ir:.0%} too low — "
        "peak load may not exceed 900 kW budget; increase Team B baseline in cluster_scenario.py"
    )


def test_reject_all_zero_incidents():
    """reject_all: no jobs admitted = no load = no power violations."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=42)
    done = False
    while not done:
        decisions = reject_all(obs)
        obs, reward, done, info = env.step(decisions)

    ir = env._grader.incident_rate()
    assert ir == 0.0, f"reject_all had incidents: {ir:.0%}"


def test_oversight_flags_appear_for_team_b():
    """Team B oversight flags (priority_inflation) should appear by window 3."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=7)
    done = False
    all_flags = []
    while not done:
        obs, reward, done, info = env.step(priority_weighted_threshold(obs))
        all_flags.extend(obs.oversight_flags)

    team_b_flags = [f for f in all_flags if f.team_id == "team_b"]
    assert len(team_b_flags) > 0, (
        "No Team B oversight flags detected — check OversightMonitor wiring "
        "in ClusterEnvironment.step()"
    )


def test_reward_bounds():
    """Per-window reward must stay within [-0.35, +0.65]."""
    env = ClusterEnvironment(enable_chiller_fault=False)
    obs = env.reset(seed=1)
    done = False
    while not done:
        decisions = priority_weighted_threshold(obs)
        obs, reward, done, info = env.step(decisions)
        assert -0.35 - 1e-6 <= reward <= 0.65 + 1e-6, f"reward {reward:.4f} out of bounds"


# ── Calibration gate ──────────────────────────────────────────────────────────


def run_calibration_gate(n_episodes: int = 10) -> dict:
    """
    Calibration gate: characterise all three scheduler tiers across N episodes.

    Tier 1 — accept_all:               incidents high (budget always blown)
    Tier 2 — priority_weighted_threshold: safe (0% incidents) but suboptimal score
    Tier 3 — trained LLM target:         safe + high throughput + high carbon deferral

    The training objective is to beat Tier 2 in total score, not just incidents.
    Returns a dict of per-tier metrics.
    """
    results: dict[str, dict] = {}

    for sched_name, sched_fn in [
        ("accept_all",                  accept_all),
        ("priority_weighted_threshold", priority_weighted_threshold),
    ]:
        env = ClusterEnvironment(enable_chiller_fault=False)
        incident_rates, scores, throughputs, carbons = [], [], [], []

        for seed in range(n_episodes):
            obs = env.reset(seed=seed)
            done = False
            while not done:
                obs, reward, done, info = env.step(sched_fn(obs))

            incident_rates.append(env._grader.incident_rate())
            scores.append(env._grader.final_score())
            throughputs.append(env._grader.mean_throughput())
            carbons.append(env._grader.mean_carbon_deferral())

        results[sched_name] = {
            "incident_rate":   sum(incident_rates) / n_episodes,
            "score":           sum(scores) / n_episodes,
            "throughput":      sum(throughputs) / n_episodes,
            "carbon_deferral": sum(carbons) / n_episodes,
        }

    print(f"\n  {'Scheduler':<32} {'Score':>8}  {'Incidents':>10}  "
          f"{'Throughput':>11}  {'CarbonDefer':>12}")
    print(f"  {'-'*32} {'-'*8}  {'-'*10}  {'-'*11}  {'-'*12}")
    for name, m in results.items():
        print(f"  {name:<32} {m['score']:>+8.4f}  {m['incident_rate']:>10.0%}  "
              f"{m['throughput']:>11.3f}  {m['carbon_deferral']:>12.3f}")

    bline = results["priority_weighted_threshold"]
    naive  = results["accept_all"]

    print(f"\n  Training targets (beat priority_weighted_threshold):")
    print(f"    Score       : >{bline['score']:+.4f}  (target >{bline['score'] + 0.10:+.4f})")
    print(f"    Incidents   : <{naive['incident_rate']:.0%}  (target <15%)")
    print(f"    Carbon defer: >{bline['carbon_deferral']:.3f}  (target >0.40)")

    # Gate: accept_all must cause incidents, baseline must be positive
    assert naive["incident_rate"] > 0.3, (
        f"accept_all incident rate {naive['incident_rate']:.0%} too low — "
        "scenario load doesn't exceed 900 kW budget; increase Team B baseline."
    )
    assert bline["score"] > 0.0, (
        f"priority_weighted_threshold score {bline['score']:+.4f} is not positive — "
        "the baseline is too conservative or throughput is broken."
    )

    return results


if __name__ == "__main__":
    import traceback

    tests = [
        ("reset returns WindowState",           test_env_reset_returns_window_state),
        ("full episode completes (8 windows)",  test_full_episode_completes),
        ("accept_all causes power violations",  test_accept_all_causes_power_violations),
        ("reject_all has zero incidents",       test_reject_all_zero_incidents),
        ("oversight flags appear for Team B",   test_oversight_flags_appear_for_team_b),
        ("per-window reward in [-0.35, +0.65]", test_reward_bounds),
    ]

    print("=" * 64)
    print("ClusterEnv Integration Gate")
    print("=" * 64)

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} unit tests passed\n")

    if passed == len(tests):
        print("Calibration gate (10 episodes, all scheduler tiers):")
        try:
            results = run_calibration_gate()
            bline = results["priority_weighted_threshold"]
            print(f"\nCALIBRATION GATE: PASS  "
                  f"(baseline score {bline['score']:+.4f}, "
                  f"accept_all incidents {results['accept_all']['incident_rate']:.0%})")
        except AssertionError as e:
            print(f"\nCALIBRATION GATE: FAIL  {e}")
    else:
        print("Skipping calibration gate — fix unit test failures first.")
