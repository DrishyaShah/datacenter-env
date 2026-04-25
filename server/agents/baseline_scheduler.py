"""
Priority-weighted threshold baseline scheduler for ClusterEnv.

This is the BASELINE policy that the GRPO-trained operator is compared against.
It is deterministic, reproducible, and reasonably sensible — representing what
a naive IT admin would implement without any learning.

Baseline behaviour (and its known failure modes):
  1. Sorts all pending requests by stated_priority (CRITICAL first).
  2. Accepts each in order if current_load + request.estimated_kw ≤ 85% capacity.
  3. Defers anything that doesn't fit to the next window.
  4. Never permanently rejects — always defers.
  5. Ignores carbon intensity — admits whenever capacity allows.
  6. Takes stated_priority at face value — gets fooled by Team B's inflation every time.
  7. Ignores oversight flags entirely.

Expected baseline metrics (verified by calibration gate):
  - Thermal incident rate: 0.40–0.65 during peak windows
  - Carbon deferral rate: ~0.04 (accidental, not intentional)
  - Throughput: ~0.41

These numbers form the "0% improvement" reference line for the GRPO training curves.

Usage:
    from server.agents.baseline_scheduler import priority_weighted_threshold
    decisions = priority_weighted_threshold(window_state)
"""

from __future__ import annotations
from server.economic import WindowState, AdmissionDecision
from server.economic.job_request import PRIORITY_ORDER


# ── Capacity safety margin ────────────────────────────────────────────────────
# The baseline accepts up to this fraction of headroom.
# 0.85 = 85% of remaining capacity — leaves 15% buffer.
# This prevents the most obvious overloads but misses the subtle ones
# (e.g. multiple 280 kW jobs each under threshold but combined over budget).

CAPACITY_THRESHOLD_FRACTION = 0.85


def priority_weighted_threshold(window_state: WindowState) -> list[AdmissionDecision]:
    """
    Deterministic baseline scheduling policy.

    Parameters
    ----------
    window_state : WindowState
        Current window's full operator observation.

    Returns
    -------
    list[AdmissionDecision]
        One decision per pending request (new + deferred).
        All requests are either ACCEPT or DEFER — never REJECT.
    """
    decisions:  list[AdmissionDecision] = []
    remaining_kw = window_state.capacity_headroom_kw

    # Sort by stated_priority descending (CRITICAL=3 first, LOW=0 last).
    # Team B's fake CRITICAL claims go to the front — this is the failure mode.
    sorted_requests = sorted(
        window_state.all_pending,
        key=lambda r: PRIORITY_ORDER.get(r.stated_priority, 0),
        reverse=True,
    )

    next_window = min(window_state.window_idx + 1, window_state.total_windows - 1)

    for req in sorted_requests:
        threshold = remaining_kw * CAPACITY_THRESHOLD_FRACTION
        if req.estimated_kw <= threshold:
            decisions.append(AdmissionDecision.accept(
                req.request_id,
                reasoning=(
                    f"accepted: {req.estimated_kw}kW fits within "
                    f"{remaining_kw:.0f}kW headroom (85% threshold)"
                ),
            ))
            remaining_kw -= req.estimated_kw
        else:
            decisions.append(AdmissionDecision.defer(
                req.request_id,
                target_window=next_window,
                reasoning=(
                    f"deferred to window {next_window}: "
                    f"{req.estimated_kw}kW exceeds 85% of {remaining_kw:.0f}kW headroom"
                ),
            ))

    return decisions


def reject_all(window_state: WindowState) -> list[AdmissionDecision]:
    """
    Reject every request — used to verify thermal incidents don't occur at zero load.
    """
    next_window = min(window_state.window_idx + 1, window_state.total_windows - 1)
    return [
        AdmissionDecision.defer(r.request_id, next_window, "reject_all baseline")
        for r in window_state.all_pending
    ]


def accept_all(window_state: WindowState) -> list[AdmissionDecision]:
    """
    Accept every request regardless of capacity — used to verify
    thermal incidents DO occur at max load (upper bound of incident rate).
    """
    return [
        AdmissionDecision.accept(r.request_id, "accept_all baseline")
        for r in window_state.all_pending
    ]
