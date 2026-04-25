"""
Operator prompt template for ClusterEnv GRPO training.

Single public function:
    build_prompt(window_state: WindowState) -> str

Converts a WindowState into the text the LLM scheduler reads each negotiation
window. All fields are sourced exclusively from WindowState — no FacilityState
or ChargebackLedger objects are passed in. The LLM then outputs a JSON block
that rollout.py parses into list[AdmissionDecision].

Prompt quality is the primary determinant of GRPO training signal quality.
Every field that matters for a correct scheduling decision must appear, clearly
labelled, in a consistent position the model can learn to locate.

JSON output schema is at the bottom of every prompt — unambiguous, no markdown
fences, no trailing text. Format errors in the LLM's response trigger a -0.5
reward penalty in rollout.py, so the schema must be impossible to misread.
"""

from __future__ import annotations

from server.economic import WindowState
from server.economic.job_request import JobRequest
from server.economic.chargeback import (
    TEAM_BUDGET_PER_EPISODE,
    BUDGET_WARNING_THRESHOLD,
    BUDGET_CRITICAL_THRESHOLD,
)


# ── Display constants ─────────────────────────────────────────────────────────

_ZONE_LABEL: dict[str, str] = {
    "green":  "GREEN  [OK]  (<23°C, safe)",
    "yellow": "YELLOW [!]   (23-25°C, warming)",
    "red":    "RED    [!!]  (>=25°C, near limit)",
}

_CARBON_LABEL: dict[str, str] = {
    "low":      "LOW      (grid mostly renewables -- good time to run flexible jobs)",
    "medium":   "MEDIUM   (grid mixed -- proceed with normal admission)",
    "high":     "HIGH     (grid peaking -- defer carbon-flexible jobs if possible)",
    "critical": "CRITICAL (grid stressed -- strongly prefer deferral of flexible jobs)",
}

_DIV = "-" * 72


# ── Private helpers ───────────────────────────────────────────────────────────


def _budget_str(remaining: float) -> str:
    """Format a team's remaining budget with warning tags."""
    pct = remaining / TEAM_BUDGET_PER_EPISODE
    if pct <= BUDGET_CRITICAL_THRESHOLD:
        tag = " [CRITICAL] -- team likely over-claiming priority"
    elif pct <= BUDGET_WARNING_THRESHOLD:
        tag = " [LOW]"
    else:
        tag = ""
    return (
        f"{remaining:.1f}/{TEAM_BUDGET_PER_EPISODE:.0f} GPU-hr equiv "
        f"({pct:.0%} remaining){tag}"
    )


def _carbon_forecast_str(forecast: list[str], window_idx: int) -> str:
    """Format the next-3-windows carbon forecast with window labels."""
    if not forecast:
        return "no forecast (final windows)"
    parts = [
        f"W{window_idx + 1 + i}: {label.upper()}"
        for i, label in enumerate(forecast)
    ]
    return " -> ".join(parts)


def _format_job(req: JobRequest, deferred: bool = False) -> str:
    """Format one job request block for the pending or deferred section."""
    tag = " [DEFERRED]" if deferred else ""
    flex = "yes" if req.stated_carbon_flexible else "no"
    return (
        f"[{req.request_id}]{tag} {req.team_id.upper()} | {req.job_type}\n"
        f'  "{req.job_description}"\n'
        f"  Power: {req.estimated_kw:.0f} kW  |  "
        f"Duration: {req.estimated_duration_hours:.0f}h  |  "
        f"Chargeback: {req.compute_budget_cost:.1f} units\n"
        f"  Priority: {req.stated_priority}  |  "
        f"Deadline: {req.stated_deadline}  |  "
        f"Carbon flexible: {flex}"
    )


# ── Public API ────────────────────────────────────────────────────────────────


def build_prompt(window_state: WindowState) -> str:
    """
    Build the full operator prompt for one negotiation window.

    Parameters
    ----------
    window_state : WindowState
        Current window's complete public observation. All private fields
        (true_priority, true_deadline_window, true_carbon_flexible) are
        absent — they live only in EpisodeLedger and are never passed here.

    Returns
    -------
    str
        The complete prompt text. Pass directly to the LLM tokenizer.
        The LLM must respond with a JSON block matching the schema at the end.
    """
    ws = window_state
    w  = ws.window_idx
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(
        f"=== AI CLUSTER SCHEDULER | WINDOW {w + 1}/{ws.total_windows}"
        f" | {ws.sim_timestamp} ==="
    )
    lines.append("")

    # ── Grid / carbon ─────────────────────────────────────────────────────────
    intensity_display = _CARBON_LABEL.get(
        ws.carbon_intensity, ws.carbon_intensity.upper()
    )
    forecast_display = _carbon_forecast_str(ws.carbon_forecast, w)
    lines.append(f"GRID: Carbon {intensity_display}")
    lines.append(f"      Forecast: {forecast_display}")
    lines.append("")

    # ── Thermal status ────────────────────────────────────────────────────────
    lines.append("THERMAL STATUS:")
    if ws.thermal_summary:
        for zone_id, status in ws.thermal_summary.items():
            label = _ZONE_LABEL.get(status, status.upper())
            lines.append(f"  {zone_id}: {label}")
    else:
        lines.append("  (no zone data)")
    lines.append("")

    # ── Capacity ──────────────────────────────────────────────────────────────
    lines.append(f"CAPACITY: {ws.capacity_headroom_kw:.0f} kW available")
    if ws.oversubscribed_if_all_accepted:
        lines.append(
            f"  [!] CAUTION: admitting all pending jobs would require "
            f"{ws.total_pending_kw:.0f} kW -- exceeds available headroom"
        )
    lines.append("")

    # ── Pending requests ──────────────────────────────────────────────────────
    lines.append(_DIV)
    lines.append("PENDING REQUESTS (new submissions this window)")
    lines.append(_DIV)
    if ws.pending_requests:
        for req in ws.pending_requests:
            lines.append(_format_job(req, deferred=False))
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    # ── Deferred requests ─────────────────────────────────────────────────────
    lines.append(_DIV)
    lines.append("DEFERRED FROM PREVIOUS WINDOWS")
    lines.append(_DIV)
    if ws.deferred_requests:
        for req in ws.deferred_requests:
            lines.append(_format_job(req, deferred=True))
            lines.append("")
    else:
        lines.append("  (none)")
        lines.append("")

    # ── Team history ──────────────────────────────────────────────────────────
    lines.append(_DIV)
    lines.append("TEAM HISTORY")
    lines.append(_DIV)
    for team_label, team_id in [("Team A (cooperative)", "team_a"),
                                  ("Team B (strategic)",   "team_b")]:
        remaining = ws.team_budgets_remaining.get(team_id, TEAM_BUDGET_PER_EPISODE)
        budget    = _budget_str(remaining)
        history   = ws.team_history.get(team_id)
        hist_str  = history.summary_str() if history else "no history yet"
        lines.append(f"{team_label}")
        lines.append(f"  Budget : {budget}")
        lines.append(f"  History: {hist_str}")
        lines.append("")

    # ── Oversight flags ───────────────────────────────────────────────────────
    prev_window = w - 1
    lines.append(_DIV)
    lines.append(
        f"OVERSIGHT FLAGS"
        + (f" (detected in window {prev_window})" if prev_window >= 0 else " (window 0 — no prior data)")
    )
    lines.append(_DIV)
    if ws.oversight_flags:
        for flag in ws.oversight_flags:
            lines.append(f"  {flag.prompt_str()}")
    else:
        lines.append("  [NONE]")
    lines.append("")

    # ── Decision instructions + JSON schema ───────────────────────────────────
    lines.append(_DIV)
    lines.append("YOUR DECISION")
    lines.append(_DIV)
    lines.append(
        "Decide for EVERY request listed above (pending + deferred): ACCEPT, DEFER, or REJECT."
    )
    lines.append("")
    lines.append("  ACCEPT  — admit now; job starts this window, power drawn immediately")
    lines.append("  DEFER   — schedule for a later window; you MUST specify scheduled_window")
    lines.append("  REJECT  — permanently decline; job is dropped")
    lines.append("")
    lines.append("Optimise for:")
    lines.append("  1. Job throughput  — complete jobs before their true deadlines")
    lines.append("  2. Power safety    — total admitted load must stay within 900 kW budget")
    lines.append("  3. Carbon efficiency — defer flexible jobs to low-carbon windows")
    lines.append("")
    lines.append(
        "Use team history and oversight flags to detect gaming. "
        "A team with high priority_inflation_rate is likely overclaiming urgency."
    )
    lines.append("")
    lines.append(
        "Output ONLY valid JSON — no markdown fences, no prose before or after:"
    )
    lines.append('{')
    lines.append('  "decisions": [')
    lines.append('    {')
    lines.append('      "request_id": "req_...",')
    lines.append('      "decision": "ACCEPT | DEFER | REJECT",')
    lines.append('      "scheduled_window": null,')
    lines.append('      "reasoning": "one sentence explaining this specific decision"')
    lines.append('    }')
    lines.append('  ],')
    lines.append('  "operator_note": "optional broadcast message to teams, or null"')
    lines.append('}')

    return "\n".join(lines)
