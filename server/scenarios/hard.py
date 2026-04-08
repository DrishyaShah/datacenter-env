"""
Hard scenario: "Cascading Failure + Carbon-Aware Triage"

Four zones:
  zone_ai_1    — critical priority, 500 kW baseline  (must not exceed 30 °C for 5+ steps)
  zone_ai_2    — critical priority, 480 kW baseline  (same constraint)
  zone_storage — medium priority,   200 kW baseline
  zone_infra   — low priority,      120 kW baseline

Fault schedule (injected via FacilityState.chiller_fault_step; rescaled by step_scale=7.2):
  Step 3  → COP begins degrading: 3.5 → 0.8 over 5 degradation steps
             (rescaled from original step 15; FAULT_START_STEP=3)
  Step 8  → Chiller goes fully offline (chiller_active = False)
             (rescaled from original step 20; CHILLER_OFFLINE_STEP=8)

After step 8 the agent has ONLY fan + free-air cooling available.  The episode
starts at 08:00 (morning ramp-up) so the chiller failure occurs during rising
outdoor temperature and peak IT load — no night-time free-cooling buffer.
Critical zones start at 25–26 °C (already stressed) and IT loads at 90 % so
the agent faces immediate thermal pressure even before the fault.

Carbon intensity follows the default 24-hr curve (condensed step references):
  Steps  0–3  : medium (08:00–09:48, morning ramp)
  Steps  4–18 : high→critical_high (10:00–16:48, midday grid demand)
  Steps 19–40 : medium→low (evening wind + overnight)

Hard termination: any critical zone > 32 °C for 5+ consecutive steps → score = 0.

Episode: 40 steps × 36 min/step = 24 hours simulated time (condensed, step_scale=7.2).
Thermal physics run at 5-min granularity; clock and load advance at 36-min/step.
Start time: 08:00 — morning ramp, immediate thermal stress.
"""

import math
from ..simulation import (
    FacilityState,
    ZoneState,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_CRITICAL,
    _default_carbon_curve,
)


# ── Scenario constants ────────────────────────────────────────────────────────

_START_HOUR = 8.0       # 08:00 — morning ramp-up; chiller fails during peak heat + load
_PID_BASELINE_PUE = 1.50   # slightly higher baseline to reflect stressed start conditions


# ── Weather curves (288 steps, 5-min resolution) ──────────────────────────────

def _hard_outside_temp_curve() -> list:
    """
    288-element outside dry-bulb temperature curve (°C).
    Realistic UK-summer 24-hr cycle: cool night → warm afternoon.

      00:00–06:00  : 16 → 14 °C  (pre-dawn dip)
      06:00–14:00  : 14 → 32 °C  (morning/afternoon rise)
      14:00–20:00  : 32 → 26 °C  (late afternoon cooling)
      20:00–24:00  : 26 → 16 °C  (night)
    """
    temps = []
    for s in range(288):
        hour = s * (5.0 / 60.0)   # convert step → fractional hour
        if hour < 6.0:
            # Pre-dawn dip: 16 → 14
            frac = hour / 6.0
            t = 16.0 - 2.0 * frac
        elif hour < 14.0:
            # Morning/afternoon rise: 14 → 32
            frac = (hour - 6.0) / 8.0
            t = 14.0 + 18.0 * math.sin(frac * math.pi / 2.0)
        elif hour < 20.0:
            # Late afternoon cooling: 32 → 26
            frac = (hour - 14.0) / 6.0
            t = 32.0 - 6.0 * frac
        else:
            # Night: 26 → 16
            frac = (hour - 20.0) / 4.0
            t = 26.0 - 10.0 * frac
        temps.append(round(t, 2))
    return temps


def _hard_wet_bulb_curve() -> list:
    """
    Wet-bulb temperature curve (°C).
    Approximately 65 % of dry-bulb rise above a 10 °C dew-point floor.
    Free cooling is viable only when wet-bulb < (zone_temp - 4 °C),
    i.e. roughly when outside is below 18 °C — steps 0–30 and 260–288.
    """
    dry = _hard_outside_temp_curve()
    return [round(10.0 + (d - 14.0) * 0.60, 2) for d in dry]


# ── IT load curve (24-hr, normalised) ────────────────────────────────────────
# AI zones carry heavy overnight inference loads plus a business-hours peak.

def _hard_load_curve() -> list:
    return [
        0.72, 0.70, 0.68, 0.67, 0.68, 0.70,   # 00–05  overnight inference
        0.75, 0.82, 0.90, 0.96, 0.99, 1.00,   # 06–11  morning ramp + peak
        1.00, 0.99, 0.97, 0.95, 0.92, 0.88,   # 12–17  sustained business peak
        0.84, 0.80, 0.78, 0.76, 0.74, 0.73,   # 18–23  evening taper
    ]


def build_hard_scenario(seed: int = 0) -> FacilityState:
    """
    Construct the initial FacilityState for the hard task.

    Chiller fault is baked in via chiller_fault_step=15.
    FacilityState.inject_chiller_fault() will automatically degrade the
    chiller starting at step 15 and take it offline at step 20.

    The seed is accepted for API compatibility; weather and load curves are
    deterministic. Sensor noise in reported_temp_c is the only stochastic
    element.
    """
    # ── zone_ai_1 — critical, primary AI inference cluster ────────────────────
    # Starts at 25 °C (already stressed at 08:00) with IT at 90 % baseline.
    # This creates immediate thermal pressure even before the chiller fault.
    zone_ai_1 = ZoneState(
        zone_id="zone_ai_1",
        temp_c=25.0,
        it_load_kw=500.0 * 0.90,       # 90 % of base at 08:00 — morning ramp already running
        fan_speed_pct=72.0,
        cooling_capacity_kw=560.0,
        setpoint_c=20.0,
        humidity_pct=46.0,
        sensor_faulty=False,
        hot_aisle_temp_c=33.5,
        supply_air_temp_c=20.0,
        supply_air_temp_setpoint_c=20.0,
        zone_priority=PRIORITY_CRITICAL,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=500.0,
        it_load_pct=0.90,
        thermal_mass_kj_per_k=944.0,   # 500/450 × 850
    )

    # ── zone_ai_2 — critical, secondary AI inference cluster ──────────────────
    zone_ai_2 = ZoneState(
        zone_id="zone_ai_2",
        temp_c=25.5,
        it_load_kw=480.0 * 0.90,
        fan_speed_pct=70.0,
        cooling_capacity_kw=540.0,
        setpoint_c=20.0,
        humidity_pct=46.0,
        sensor_faulty=False,
        hot_aisle_temp_c=34.0,
        supply_air_temp_c=20.0,
        supply_air_temp_setpoint_c=20.0,
        zone_priority=PRIORITY_CRITICAL,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=480.0,
        it_load_pct=0.90,
        thermal_mass_kj_per_k=907.0,   # 480/450 × 850
    )

    # ── zone_storage — medium priority ────────────────────────────────────────
    # Warmer at start — agent may need to sacrifice this zone for the AI clusters.
    zone_storage = ZoneState(
        zone_id="zone_storage",
        temp_c=25.0,
        it_load_kw=200.0 * 0.90,
        fan_speed_pct=58.0,
        cooling_capacity_kw=230.0,
        setpoint_c=22.0,
        humidity_pct=49.0,
        sensor_faulty=False,
        hot_aisle_temp_c=33.0,
        supply_air_temp_c=22.0,
        supply_air_temp_setpoint_c=22.0,
        zone_priority=PRIORITY_MEDIUM,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=200.0,
        it_load_pct=0.90,
        thermal_mass_kj_per_k=378.0,   # 200/450 × 850
    )

    # ── zone_infra — low priority, monitoring / logging ───────────────────────
    # Cheapest to sacrifice. Higher starting temp and reduced fans create
    # immediate tension, forcing triage decisions from step 1.
    zone_infra = ZoneState(
        zone_id="zone_infra",
        temp_c=26.0,
        it_load_kw=120.0 * 0.90,
        fan_speed_pct=50.0,
        cooling_capacity_kw=140.0,
        setpoint_c=23.0,
        humidity_pct=50.0,
        sensor_faulty=False,
        hot_aisle_temp_c=34.5,
        supply_air_temp_c=23.0,
        supply_air_temp_setpoint_c=23.0,
        zone_priority=PRIORITY_LOW,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=120.0,
        it_load_pct=0.90,
        thermal_mass_kj_per_k=227.0,   # 120/450 × 850
    )

    outside_temps = _hard_outside_temp_curve()
    wet_bulbs     = _hard_wet_bulb_curve()

    # Index into the weather curves at the start hour so the episode begins at
    # the right point in the 24-hr diurnal cycle.
    _start_idx = int(_START_HOUR * 60 / 5)   # 5-min steps from midnight
    _start_idx = min(_start_idx, len(outside_temps) - 1)

    facility = FacilityState(
        zones=[zone_ai_1, zone_ai_2, zone_storage, zone_infra],
        outside_temp_c=outside_temps[_start_idx],   # ~20 °C at 08:00
        # chiller — healthy at start; fault injected at step 15 (rescaled → step 3)
        chiller_active=True,
        chiller_cop=3.5,
        chiller_setpoint_c=10.0,
        chiller_fault_step=15,                       # ← raw fault step (rescaled in reset())
        chiller_fault_level=0.0,
        # weather — starting at 08:00 in the diurnal cycle
        wet_bulb_temp_c=wet_bulbs[_start_idx],
        # time
        timestamp_hour=_START_HOUR,
        step_number=0,
        # power
        ups_efficiency=0.96,
        # carbon — 08:00 is already in the medium-carbon morning ramp (≈0.60)
        grid_carbon_curve=_default_carbon_curve(),
        grid_carbon_intensity="medium",
        grid_carbon_intensity_normalized=0.60,   # 08:00 value from default curve
        # load — inference load already at 90 % at 08:00
        load_curve=_hard_load_curve(),
        # grader reference
        pid_baseline_pue=_PID_BASELINE_PUE,
        # context
        maintenance_notes=[],
        upcoming_events=[
            "CRITICAL: Chiller health monitoring detected an anomaly at 07:45. "
            "COP degradation is imminent — expect fault within 2–3 steps.",
            "Carbon intensity: MEDIUM now (08:00), rising to CRITICAL_HIGH "
            "between 10:00–16:00, easing to MEDIUM by 20:00.",
            "zone_ai_1 and zone_ai_2 are at 90 % load running active inference. "
            "SLA requires cold-aisle temp ≤ 30 °C at all times.",
            "zone_infra can tolerate up to 29 °C for short periods without data loss.",
            "Free-cooling via economiser is available only when outside temp < 18 °C. "
            "Outside is currently ~20 °C — free cooling unavailable until tonight.",
        ],
    )

    # Attach per-step weather curves starting from _start_idx so environment.py
    # can update outside_temp_c each step relative to the episode start.
    facility._outside_temp_curve = outside_temps[_start_idx:]
    facility._wet_bulb_curve     = wet_bulbs[_start_idx:]

    return facility