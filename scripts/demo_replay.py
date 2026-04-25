"""
ClusterEnv demo recording and replay tool.

Usage:
    # Generate baseline episode recording
    python scripts/demo_replay.py --generate --output demo_baseline.json

    # Replay a recording with formatted output
    python scripts/demo_replay.py demo_baseline.json

    # Compare two recordings side by side
    python scripts/demo_replay.py demo_baseline.json --compare demo_trained.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.cluster_environment import ClusterEnvironment
from server.agents.baseline_scheduler import priority_weighted_threshold
from server.economic.job_request import AdmissionDecision


# ── Display helpers ───────────────────────────────────────────────────────────

_THERMAL_COLOR = {
    "green":  "OK   (<23C)",
    "yellow": "WARM (23-25C)",
    "red":    "HOT  (>=25C)",
}

_CARBON_SYMBOL = {
    "low":      "[LOW]",
    "medium":   "[MED]",
    "high":     "[HIGH]",
    "critical": "[CRIT]",
}

DIV = "-" * 68


def _fmt_decisions(decisions: list[dict]) -> str:
    lines = []
    for d in decisions:
        sw = f" -> W{d['scheduled_window']}" if d["scheduled_window"] is not None else ""
        lines.append(f"    {d['request_id']}: {d['decision']}{sw}  ({d['reasoning']})")
    return "\n".join(lines) if lines else "    (none)"


def _fmt_flags(flags: list[dict]) -> str:
    if not flags:
        return "    [NONE]"
    lines = []
    for f in flags:
        lines.append(
            f"    [{f['severity'].upper()}] {f['team_id']}: {f['flag_type']} "
            f"(conf {f['confidence']:.0%})"
        )
    return "\n".join(lines)


def replay(recording: dict, label: str = "") -> None:
    """Pretty-print a full episode recording."""
    meta = recording.get("meta", {})
    windows = recording.get("windows", [])

    print(DIV)
    header = "CLUSTERENV EPISODE REPLAY"
    if label:
        header += f" — {label}"
    print(header)
    print(DIV)
    print(f"  Scheduler    : {meta.get('scheduler', 'unknown')}")
    print(f"  Generated at : {meta.get('generated_at', 'unknown')}")
    print(f"  Final score  : {meta.get('final_score', 0.0):+.4f}")
    print(f"  Incident rate: {meta.get('incident_rate', 0.0):.0%}")
    print(f"  Throughput   : {meta.get('mean_throughput', 0.0):.2f}")
    print(f"  Carbon defer : {meta.get('mean_carbon_deferral', 0.0):.2f}")
    print()

    for w in windows:
        idx    = w["window_idx"]
        carbon = w.get("carbon_intensity", "?")
        print(f"  Window {idx + 1}/8  |  {w.get('sim_timestamp', '?')}  |  "
              f"Carbon {_CARBON_SYMBOL.get(carbon, carbon.upper())}  |  "
              f"Reward {w['reward']:+.4f}  |  "
              f"{'[INCIDENT]' if w['power_violated'] else '[OK]'}")

        # Thermal
        thermal = w.get("thermal_summary", {})
        if thermal:
            parts = [f"{zid}: {_THERMAL_COLOR.get(s, s)}" for zid, s in thermal.items()]
            print(f"    Thermal: {' | '.join(parts)}")

        # Decisions
        print(f"    Decisions ({len(w.get('decisions', []))} jobs):")
        print(_fmt_decisions(w.get("decisions", [])))

        # Flags
        flags = w.get("oversight_flags_next_window", [])
        if flags:
            print(f"    Oversight flags (-> next window):")
            print(_fmt_flags(flags))

        print()

    print(DIV)
    print(f"Final score: {meta.get('final_score', 0.0):+.4f}  |  "
          f"Incidents: {meta.get('incident_rate', 0.0):.0%}  |  "
          f"Carbon deferral: {meta.get('mean_carbon_deferral', 0.0):.2f}")
    print(DIV)


def compare(baseline: dict, trained: dict) -> None:
    """Side-by-side comparison of two episode recordings."""
    print(DIV)
    print("CLUSTERENV — BASELINE vs. TRAINED COMPARISON")
    print(DIV)

    metrics = [
        ("Final score",       "final_score",          "{:+.4f}",  True),
        ("Incident rate",     "incident_rate",         "{:.0%}",   False),
        ("Throughput",        "mean_throughput",       "{:.3f}",   True),
        ("Carbon deferral",   "mean_carbon_deferral",  "{:.3f}",   True),
    ]

    b_meta = baseline.get("meta", {})
    t_meta = trained.get("meta", {})

    print(f"  {'Metric':<22} {'Baseline':>12}  {'Trained':>12}  {'Delta':>10}")
    print(f"  {'-'*22} {'-'*12}  {'-'*12}  {'-'*10}")

    for label, key, fmt, higher_is_better in metrics:
        bval = b_meta.get(key, 0.0)
        tval = t_meta.get(key, 0.0)
        delta = tval - bval
        sign  = "+" if delta > 0 else ""
        arrow = "^" if (delta > 0) == higher_is_better else "v"
        print(f"  {label:<22} {fmt.format(bval):>12}  {fmt.format(tval):>12}  "
              f"{sign}{delta:+.4f} {arrow}")

    # Per-window incident comparison
    print()
    print(f"  {'Window':<10} {'Baseline':>12}  {'Trained':>12}")
    print(f"  {'-'*10} {'-'*12}  {'-'*12}")
    b_wins = {w["window_idx"]: w for w in baseline.get("windows", [])}
    t_wins = {w["window_idx"]: w for w in trained.get("windows", [])}
    for i in range(8):
        bw = b_wins.get(i, {})
        tw = t_wins.get(i, {})
        b_inc = "[INCIDENT]" if bw.get("power_violated") else "[OK]"
        t_inc = "[INCIDENT]" if tw.get("power_violated") else "[OK]"
        print(f"  W{i + 1:<9} {b_inc:>12}  {t_inc:>12}")

    print(DIV)


# ── Generation ────────────────────────────────────────────────────────────────


def generate_episode(
    scheduler_name: str = "priority_weighted_threshold",
    seed: int = 42,
    enable_chiller_fault: bool = False,
) -> dict:
    """Run one full episode and return a serialisable recording dict."""
    env = ClusterEnvironment(enable_chiller_fault=enable_chiller_fault)
    obs = env.reset(seed=seed)
    done = False
    window_records = []

    while not done:
        # Build decisions
        if scheduler_name == "priority_weighted_threshold":
            decisions = priority_weighted_threshold(obs)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        # Record pre-step state
        pre_thermal  = dict(obs.thermal_summary)
        pre_carbon   = obs.carbon_intensity
        pre_headroom = obs.capacity_headroom_kw
        pre_pending  = [r.public_fields() for r in obs.all_pending]
        w_idx        = obs.window_idx
        w_ts         = obs.sim_timestamp

        obs, reward, done, info = env.step(decisions)

        # Flags injected into NEXT window come from current obs after step
        next_flags = [
            {
                "team_id":        f.team_id,
                "flag_type":      f.flag_type,
                "severity":       f.severity,
                "confidence":     f.confidence,
                "evidence":       f.evidence,
                "window_detected": f.window_detected,
            }
            for f in obs.oversight_flags
        ]

        window_records.append({
            "window_idx":       w_idx,
            "sim_timestamp":    w_ts,
            "carbon_intensity": pre_carbon,
            "thermal_summary":  pre_thermal,
            "capacity_headroom_kw": pre_headroom,
            "pending_requests": pre_pending,
            "decisions": [d.to_dict() for d in decisions],
            "reward":           reward,
            "power_violated":   info.get("power_violated", False),
            "jobs_admitted":    info.get("jobs_admitted", 0),
            "jobs_on_time":     info.get("jobs_completed_on_time", 0),
            "oversight_flags_next_window": next_flags,
        })

    grader = env._grader
    return {
        "meta": {
            "scheduler":           scheduler_name,
            "seed":                seed,
            "generated_at":        datetime.now().isoformat(timespec="seconds"),
            "final_score":         grader.final_score(),
            "incident_rate":       grader.incident_rate(),
            "mean_throughput":     grader.mean_throughput(),
            "mean_carbon_deferral": grader.mean_carbon_deferral(),
        },
        "windows": window_records,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ClusterEnv demo recorder/replayer")
    parser.add_argument("recording", nargs="?", help="Path to a JSON recording to replay")
    parser.add_argument("--generate",  action="store_true", help="Generate a new recording")
    parser.add_argument("--scheduler", default="priority_weighted_threshold",
                        help="Scheduler to use when generating (default: priority_weighted_threshold)")
    parser.add_argument("--seed",      type=int, default=42, help="Episode seed (default: 42)")
    parser.add_argument("--output",    default="demo_baseline.json",
                        help="Output file for --generate mode (default: demo_baseline.json)")
    parser.add_argument("--compare",   help="Second recording to compare against")
    args = parser.parse_args()

    if args.generate:
        print(f"Generating episode (scheduler={args.scheduler}, seed={args.seed}) ...")
        recording = generate_episode(
            scheduler_name=args.scheduler,
            seed=args.seed,
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(recording, f, indent=2)
        print(f"Saved -> {args.output}")
        print()
        replay(recording, label=args.scheduler)
        return

    if args.recording is None:
        parser.print_help()
        sys.exit(1)

    with open(args.recording, encoding="utf-8") as f:
        recording = json.load(f)

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            trained = json.load(f)
        compare(recording, trained)
    else:
        replay(recording, label=os.path.basename(args.recording))


if __name__ == "__main__":
    main()
