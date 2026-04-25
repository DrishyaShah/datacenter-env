"""
ClusterEnv scenario configuration — cluster_hard.

8 negotiation windows × 18 physical steps = 144 physical steps ≈ 12 simulated hours.
Covers 08:00–18:30 on a hot summer weekday.

Zone layout:
  zone_team_a_1  — Team A training zone (starts empty, filled by admitted jobs)
  zone_team_a_2  — Team A overflow zone  (starts empty, filled by admitted jobs)
  zone_team_b_1  — Team B inference zone (180 kW baseline always on + admitted extras)
  zone_shared    — Shared infrastructure (100 kW baseline always on)

Thermal incident mechanism:
  Individual zone cooling capacities (480–500 kW) are sufficient for individual jobs.
  Thermal incidents occur through two coupled mechanisms:

  1. Chiller fault at window 5 (enabled by default):
     Chiller degrades over 5 physical steps then goes offline. Supply air temperature
     rises to wet-bulb + 2°C (~17–19°C depending on window). With reduced delta-T
     between hot-aisle and supply, effective cooling drops sharply. Zones running
     large jobs (300–320 kW) cannot shed heat fast enough → temp exceeds 27°C.

  2. Elevated starting temperatures (23.5–24.5°C) + high outdoor temp (32°C):
     Pre-fault windows 2–4 have worst outdoor temps. Zones admitting large jobs
     start warm and approach 27°C during the 18 physical steps before fans fully ramp.

  Rule-based baseline failure path:
     Accepts Team B's CRITICAL jobs (true priority MEDIUM) before window 5 chiller fault.
     Active 300+ kW jobs in zone_team_b_1 when chiller fails → incidents in windows 5–7.
     Trained scheduler should learn to reduce admitted load before window 5 fault.

Calibration gate (run before any LLM training):
  Run priority_weighted_threshold scheduler for 20 episodes.
  Assert 0.40 ≤ incident_rate ≤ 0.65 across windows 4–7.
  If below 0.40: increase Team B's extra job sizes or raise starting temperatures.
  If above 0.65: reduce starting temperatures or widen zone cooling capacities.

Calibration gate (run before any LLM training):
  Run priority_weighted_threshold scheduler for 20 episodes.
  Assert 0.40 ≤ incident_rate ≤ 0.65 across windows 2–4.
  If below 0.40: increase TEAM_B_INFERENCE_BASELINE_KW or peak job sizes.
  If above 0.65: reduce peak job sizes or widen TOTAL_POWER_BUDGET_KW.
"""

from __future__ import annotations
import math
from typing import Optional

from server.simulation import (
    FacilityState,
    ZoneState,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_CRITICAL,
    _default_carbon_curve,
)


# ── Facility constants ────────────────────────────────────────────────────────

TOTAL_POWER_BUDGET_KW      = 900.0    # hard facility limit; scheduler enforces via admission
WINDOWS_PER_EPISODE        = 8
PHYSICAL_STEPS_PER_WINDOW  = 18       # 18 × 5 min = 90 sim-minutes per window
WINDOW_DURATION_HOURS      = 1.5      # 90 sim-minutes → 1.5 hours
START_HOUR                 = 8.0      # episode begins at 08:00

# Zone sizing
TEAM_A_ZONE_CAPACITY_KW    = 480.0    # matches existing Round 1 physics calibration
TEAM_B_ZONE_CAPACITY_KW    = 500.0
SHARED_ZONE_CAPACITY_KW    = 300.0

# Zone baselines (always-on load, regardless of admitted jobs)
TEAM_A_BASELINE_KW         = 0.0      # Team A zones start empty
TEAM_B_INFERENCE_BASELINE_KW = 180.0  # Team B inference always running
SHARED_BASELINE_KW         = 100.0    # Infrastructure always running

# Starting temperatures — moderately stressed to match late-morning busy-hours.
# These are intentionally warmer than the Round 1 scenarios so that admitted
# job loads, combined with rising outdoor temps and the window-5 chiller fault,
# produce 40–65% thermal incidents with the rule-based baseline.
TEAM_A_START_TEMP_C        = 23.5    # warm from morning workloads
TEAM_B_START_TEMP_C        = 24.5    # inference load keeps this zone warmer
SHARED_START_TEMP_C        = 22.5

# Starting fan speeds (%)
DEFAULT_FAN_SPEED_PCT      = 60.0
TEAM_B_FAN_SPEED_PCT       = 68.0     # slightly higher due to continuous baseline load


# ── Per-window schedules (index = window 0..7) ────────────────────────────────

# Carbon intensity label per window (drives carbon deferral reward signal)
# LOW morning → HIGH midday peak → LOW evening is the key signal for training
CARBON_SCHEDULE: list[str] = [
    "low",      # window 0 — 08:00: overnight renewables still dominant
    "low",      # window 1 — 09:30: grid starting to ramp
    "high",     # window 2 — 11:00: morning demand surge
    "high",     # window 3 — 12:30: peak grid demand
    "high",     # window 4 — 14:00: peak sustained
    "medium",   # window 5 — 15:30: afternoon taper begins
    "low",      # window 6 — 17:00: evening wind picks up
    "low",      # window 7 — 18:30: low-carbon window — ideal for deferred batch jobs
]

# Outside dry-bulb temperature per window (°C) — affects chiller COP and free-cooling
OUTSIDE_TEMP_SCHEDULE: list[float] = [
    18.0,   # window 0 — 08:00: cool morning
    22.0,   # window 1 — 09:30: warming up
    28.0,   # window 2 — 11:00: getting hot
    32.0,   # window 3 — 12:30: peak heat → worst COP
    32.0,   # window 4 — 14:00: still peak
    29.0,   # window 5 — 15:30: slight cooling
    24.0,   # window 6 — 17:00: afternoon cooling
    19.0,   # window 7 — 18:30: evening cool
]

# Wet-bulb temperature per window (°C) — determines free-cooling potential
WET_BULB_SCHEDULE: list[float] = [
    14.0, 16.0, 20.0, 23.0, 23.0, 21.0, 18.0, 15.0
]

# Carbon intensity numeric [0–1] — derived from CARBON_SCHEDULE for reward computation
_CARBON_NUMERIC: dict[str, float] = {
    "low": 0.20, "medium": 0.55, "high": 0.82, "critical": 0.92
}

CARBON_NUMERIC_SCHEDULE: list[float] = [
    _CARBON_NUMERIC[label] for label in CARBON_SCHEDULE
]

# Peak demand windows (for calibration gate verification)
PEAK_DEMAND_WINDOWS: list[int] = [2, 3, 4]


# ── Simulated clock helpers ───────────────────────────────────────────────────

def window_to_timestamp(window_idx: int) -> str:
    """Convert window index to simulated wall-clock string (e.g. '08:00', '10:30')."""
    total_minutes = int(START_HOUR * 60) + window_idx * int(WINDOW_DURATION_HOURS * 60)
    hours   = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

def window_to_hour(window_idx: int) -> float:
    """Convert window index to simulated hour [0–24]."""
    return START_HOUR + window_idx * WINDOW_DURATION_HOURS


# ── Facility builder ──────────────────────────────────────────────────────────

def build_cluster_facility(
    seed: Optional[int] = None,
    window_idx: int = 0,
    enable_chiller_fault: bool = True,
    chiller_fault_window: int = 5,
) -> FacilityState:
    """
    Build a FacilityState configured for the cluster_hard scenario.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible episode starts.
    window_idx : int
        Starting negotiation window (0–7). Sets initial temperatures, outside
        temp, and carbon intensity to match that window's conditions.
        Default 0 = start at 08:00 with cool morning conditions.
    enable_chiller_fault : bool
        If True, a chiller fault is injected at chiller_fault_window.
        Disabled by default for the standard cluster scenario.
    chiller_fault_window : int
        Negotiation window at which chiller fault begins (if enabled).
        Translated to physical step: chiller_fault_window × PHYSICAL_STEPS_PER_WINDOW.
    """
    import random as _random
    if seed is not None:
        _random.seed(seed)

    outside_temp  = OUTSIDE_TEMP_SCHEDULE[window_idx]
    wet_bulb_temp = WET_BULB_SCHEDULE[window_idx]

    # ── Zone definitions ──────────────────────────────────────────────────────

    zones = [
        # Team A — training workloads, starts completely empty
        ZoneState(
            zone_id              = "zone_team_a_1",
            temp_c               = TEAM_A_START_TEMP_C,
            it_load_kw           = TEAM_A_BASELINE_KW,
            fan_speed_pct        = DEFAULT_FAN_SPEED_PCT,
            cooling_capacity_kw  = TEAM_A_ZONE_CAPACITY_KW,
            base_it_load_kw      = TEAM_A_BASELINE_KW,
            zone_priority        = PRIORITY_MEDIUM,
            supply_air_temp_setpoint_c = 22.0,
            thermal_mass_kj_per_k      = 850.0,
        ),
        ZoneState(
            zone_id              = "zone_team_a_2",
            temp_c               = TEAM_A_START_TEMP_C,
            it_load_kw           = TEAM_A_BASELINE_KW,
            fan_speed_pct        = DEFAULT_FAN_SPEED_PCT,
            cooling_capacity_kw  = TEAM_A_ZONE_CAPACITY_KW,
            base_it_load_kw      = TEAM_A_BASELINE_KW,
            zone_priority        = PRIORITY_MEDIUM,
            supply_air_temp_setpoint_c = 22.0,
            thermal_mass_kj_per_k      = 850.0,
        ),
        # Team B — inference always running; admitted extras add on top
        ZoneState(
            zone_id              = "zone_team_b_1",
            temp_c               = TEAM_B_START_TEMP_C,
            it_load_kw           = TEAM_B_INFERENCE_BASELINE_KW,
            fan_speed_pct        = TEAM_B_FAN_SPEED_PCT,
            cooling_capacity_kw  = TEAM_B_ZONE_CAPACITY_KW,
            base_it_load_kw      = TEAM_B_INFERENCE_BASELINE_KW,
            zone_priority        = PRIORITY_CRITICAL,
            supply_air_temp_setpoint_c = 20.0,   # inference zone kept cooler
            thermal_mass_kj_per_k      = 900.0,  # slightly higher mass (denser rack)
        ),
        # Shared infrastructure — always on, not a target for job admission
        ZoneState(
            zone_id              = "zone_shared",
            temp_c               = SHARED_START_TEMP_C,
            it_load_kw           = SHARED_BASELINE_KW,
            fan_speed_pct        = DEFAULT_FAN_SPEED_PCT,
            cooling_capacity_kw  = SHARED_ZONE_CAPACITY_KW,
            base_it_load_kw      = SHARED_BASELINE_KW,
            zone_priority        = PRIORITY_LOW,
            supply_air_temp_setpoint_c = 22.0,
            thermal_mass_kj_per_k      = 600.0,  # lighter infrastructure racks
        ),
    ]

    # ── Chiller fault config ──────────────────────────────────────────────────
    fault_step = -1  # -1 = no fault
    if enable_chiller_fault:
        fault_step = chiller_fault_window * PHYSICAL_STEPS_PER_WINDOW

    # ── Facility assembly ─────────────────────────────────────────────────────
    facility = FacilityState(
        zones              = zones,
        outside_temp_c     = outside_temp,
        wet_bulb_temp_c    = wet_bulb_temp,
        timestamp_hour     = window_to_hour(window_idx),
        chiller_active     = True,
        chiller_cop        = 3.5,
        chiller_setpoint_c = 10.0,
        chiller_fault_step = fault_step,
        ups_efficiency     = 0.96,
        minutes_per_step   = 5.0,          # always 5-min physical steps in cluster mode
        cluster_mode       = True,
        grid_carbon_intensity             = CARBON_SCHEDULE[window_idx],
        grid_carbon_intensity_normalized  = CARBON_NUMERIC_SCHEDULE[window_idx],
    )

    return facility


# ── Zone assignment helpers ───────────────────────────────────────────────────

# Which zone a team's admitted job runs in.
# Team A alternates between its two zones based on current load.
# Team B always uses its dedicated zone.
TEAM_ZONE_MAP: dict[str, list[str]] = {
    "team_a": ["zone_team_a_1", "zone_team_a_2"],
    "team_b": ["zone_team_b_1"],
}

def assign_zone(team_id: str, facility: FacilityState) -> str:
    """
    Pick the cooler / less-loaded Team A zone for job placement.
    For Team B, always returns zone_team_b_1.
    """
    candidate_ids = TEAM_ZONE_MAP.get(team_id, [])
    if not candidate_ids:
        return "zone_shared"

    if len(candidate_ids) == 1:
        return candidate_ids[0]

    # Pick the zone with lower current temperature (less thermal stress)
    zone_map = {z.zone_id: z for z in facility.zones}
    best = min(
        candidate_ids,
        key=lambda zid: zone_map[zid].temp_c if zid in zone_map else 99.0,
    )
    return best


# ── Capacity accounting ───────────────────────────────────────────────────────

def compute_headroom_kw(facility: FacilityState) -> float:
    """
    Remaining power budget = TOTAL_POWER_BUDGET_KW minus all currently
    active loads (baselines + admitted job loads).
    """
    return max(0.0, TOTAL_POWER_BUDGET_KW - facility.total_it_load_kw)


def thermal_summary(facility: FacilityState) -> dict[str, str]:
    """
    Coarse per-zone thermal status for the operator prompt.
    green:  temp < 23°C
    yellow: 23°C ≤ temp < 25°C
    red:    temp ≥ 25°C
    """
    summary = {}
    for zone in facility.zones:
        if zone.temp_c < 23.0:
            summary[zone.zone_id] = "green"
        elif zone.temp_c < 25.0:
            summary[zone.zone_id] = "yellow"
        else:
            summary[zone.zone_id] = "red"
    return summary


def any_zone_overheated(facility: FacilityState, threshold_c: float = 27.0) -> bool:
    """True if any zone exceeded the thermal incident threshold."""
    return any(z.temp_c > threshold_c for z in facility.zones)


def power_budget_violated(facility: FacilityState) -> bool:
    """
    True if total admitted IT load exceeds the facility power budget.

    ── WHY THIS IS THE PRIMARY INCIDENT METRIC FOR CLUSTERENV ──────────────────
    The thermal physics (COOLING_DELTA_T_REF = 9.0, delta_temp clamped to ±2°C)
    is calibrated for the Round 1 use case where the LLM controls fan speeds.
    In that context the agent can cause overheating by setting fans too low.

    In ClusterEnv the CoolingHeuristic always makes locally optimal cooling
    decisions, so individual zones never overheat regardless of admitted load
    (the delta_T physics self-corrects as long as each zone has any cooling
    headroom). Temperature-based incidents would require loads far exceeding
    zone capacity (>600 kW in a 480 kW zone), which is unrealistic for the
    job sizes in the scenario archetypes.

    Power budget violation is the physically correct incident for cluster
    scheduling: PDUs enforce hard power caps via circuit breakers, not
    temperature sensors. Exceeding TOTAL_POWER_BUDGET_KW causes breaker trips
    → load shedding → job kills → SLA failures. The scheduler's entire job is
    to prevent this through intelligent admission control.

    ── CALIBRATION ─────────────────────────────────────────────────────────────
    With the accept_all baseline (all jobs admitted regardless of budget):
      Peak windows (2–4): total load ≈ 1100–1300 kW >> 900 kW → 100% violation rate
    With priority_weighted_threshold (85% capacity threshold):
      Total load ≈ 700–800 kW < 900 kW → ~0% violation rate
    With trained scheduler (GRPO):
      Should learn to selectively admit: 10–20% violation rate while
      completing more Team A jobs (better throughput than rule-based).

    The demo story: accept_all (100% violation) vs trained (≤15% violation).
    ────────────────────────────────────────────────────────────────────────────
    """
    return facility.total_it_load_kw > TOTAL_POWER_BUDGET_KW


def power_overrun_kw(facility: FacilityState) -> float:
    """How many kW over budget the current admission is. 0.0 if within budget."""
    return max(0.0, facility.total_it_load_kw - TOTAL_POWER_BUDGET_KW)
