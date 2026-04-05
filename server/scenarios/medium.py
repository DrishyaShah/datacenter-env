"""
Medium scenario: "Multi-Zone Load Surge with Sensor Fault"

Three zones:
  zone_ai      — critical priority, 600 kW baseline, FAULTY sensor (+3 °C drift,
                 growing to +12 °C by step 50; sensor_confidence degrades 1.0 → 0.1)
  zone_storage — medium priority, 200 kW baseline
  zone_infra   — low priority, 150 kW baseline

IT load follows a diurnal curve that surges from ~60 % → ~95 % between
steps 30–60 (morning batch window, starting at 06:00).
Outside temperature varies: starts at 18 °C (night), peaks at 34 °C around
step 96 (noon).

Episode: 144 steps × 5 min = 12 hours simulated time.
Start time: 06:00 — diurnal ramp-up begins immediately.

Hard termination: any zone unsafe for 3+ consecutive steps → episode ends.
"""

import math
from simulation import (
    FacilityState,
    ZoneState,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_CRITICAL,
    _default_carbon_curve,
)


# ── Scenario constants ────────────────────────────────────────────────────────

_START_HOUR = 6.0       # 06:00 — diurnal ramp-up begins at step 0
_PID_BASELINE_PUE = 1.52


# ── Outside temperature curve (per-step, 144 steps, 5-min resolution) ────────
# Starts at 18 °C at 06:00, rises to 34 °C around 12:00 (step 72),
# then falls back toward 28 °C by 18:00 (step 144).
# Generated analytically so it is deterministic.

def _medium_outside_temp_curve() -> list:
    """
    144-element list of outside dry-bulb temperatures (°C), one per step.
    Steps 0–71: 18 → 34 °C (sine ramp, morning to noon).
    Steps 72–143: 34 → 28 °C (gradual afternoon cooling).
    """
    temps = []
    for s in range(144):
        if s < 72:
            # 0 → π/2: smooth rise 18 → 34
            frac = s / 72.0
            t = 18.0 + 16.0 * math.sin(frac * math.pi / 2.0)
        else:
            # π/2 → π: 34 → 28, slower decline
            frac = (s - 72) / 72.0
            t = 34.0 - 6.0 * math.sin(frac * math.pi / 2.0)
        temps.append(round(t, 2))
    return temps


# Corresponding wet-bulb curve (roughly 60 % of dry-bulb rise)
def _medium_wet_bulb_curve() -> list:
    dry = _medium_outside_temp_curve()
    return [round(14.0 + (d - 18.0) * 0.55, 2) for d in dry]


# ── IT load curve (24-hr, normalised) ────────────────────────────────────────
# Aggressive morning ramp: 60 % at 06:00, 95 % by 10:00, holds through noon.

def _medium_load_curve() -> list:
    return [
        0.50, 0.48, 0.47, 0.46, 0.48, 0.55,   # 00–05  night
        0.60, 0.72, 0.83, 0.92, 0.95, 0.97,   # 06–11  morning surge (agent's main challenge)
        0.98, 0.97, 0.95, 0.93, 0.88, 0.82,   # 12–17  sustained peak
        0.75, 0.68, 0.63, 0.58, 0.54, 0.51,   # 18–23  evening taper
    ]


def build_medium_scenario(seed: int = 0) -> FacilityState:
    """
    Construct the initial FacilityState for the medium task.

    The seed is accepted for API compatibility.  Outside temperature and
    load follow deterministic curves; sensor noise is the only stochastic
    element.
    """
    # ── zone_ai — critical, faulty sensor ────────────────────────────────────
    # Starts comfortably cool (22 °C) so the agent cannot trivially detect
    # the fault from the very first reading; it must notice the discrepancy
    # between reported_temp and physics over time.
    zone_ai = ZoneState(
        zone_id="zone_ai",
        temp_c=22.0,
        it_load_kw=600.0,
        fan_speed_pct=65.0,
        cooling_capacity_kw=680.0,
        setpoint_c=20.0,
        humidity_pct=45.0,
        sensor_faulty=True,         # FAULTY — drift starts at +3 °C, grows to +12 °C
        hot_aisle_temp_c=30.0,
        supply_air_temp_c=20.0,
        supply_air_temp_setpoint_c=20.0,
        zone_priority=PRIORITY_CRITICAL,
        sensor_drift_c=3.0,         # already drifted +3 °C at episode start
        sensor_confidence=0.77,     # 1.0 - 3.0/13.0 ≈ 0.77
        base_it_load_kw=600.0,
        it_load_pct=0.60,           # load at 60 % at 06:00
    )

    # ── zone_storage — medium priority, healthy sensor ────────────────────────
    zone_storage = ZoneState(
        zone_id="zone_storage",
        temp_c=21.5,
        it_load_kw=200.0 * 0.60,    # 60 % of base at 06:00
        fan_speed_pct=50.0,
        cooling_capacity_kw=240.0,
        setpoint_c=21.0,
        humidity_pct=47.0,
        sensor_faulty=False,
        hot_aisle_temp_c=29.5,
        supply_air_temp_c=21.0,
        supply_air_temp_setpoint_c=21.0,
        zone_priority=PRIORITY_MEDIUM,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=200.0,
        it_load_pct=0.60,
    )

    # ── zone_infra — low priority, healthy sensor ─────────────────────────────
    zone_infra = ZoneState(
        zone_id="zone_infra",
        temp_c=21.0,
        it_load_kw=150.0 * 0.60,
        fan_speed_pct=45.0,
        cooling_capacity_kw=180.0,
        setpoint_c=22.0,
        humidity_pct=46.0,
        sensor_faulty=False,
        hot_aisle_temp_c=29.0,
        supply_air_temp_c=22.0,
        supply_air_temp_setpoint_c=22.0,
        zone_priority=PRIORITY_LOW,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=150.0,
        it_load_pct=0.60,
    )

    outside_temps = _medium_outside_temp_curve()
    wet_bulbs = _medium_wet_bulb_curve()

    facility = FacilityState(
        zones=[zone_ai, zone_storage, zone_infra],
        outside_temp_c=outside_temps[0],       # 18 °C at 06:00
        # chiller — healthy, no fault
        chiller_active=True,
        chiller_cop=3.5,
        chiller_setpoint_c=10.0,
        chiller_fault_step=-1,
        chiller_fault_level=0.0,
        # weather
        wet_bulb_temp_c=wet_bulbs[0],
        # time
        timestamp_hour=_START_HOUR,
        step_number=0,
        # power
        ups_efficiency=0.96,
        # carbon
        grid_carbon_curve=_default_carbon_curve(),
        grid_carbon_intensity="low",
        grid_carbon_intensity_normalized=0.30,  # 06:00 value from default curve
        # load — aggressive morning surge
        load_curve=_medium_load_curve(),
        # grader reference
        pid_baseline_pue=_PID_BASELINE_PUE,
        # context
        maintenance_notes=[],
        upcoming_events=[
            "Morning batch job window starts at 07:00 — expect zone_ai load to surge "
            "from 60 % to 95 % over the next 2 hours.",
            "Outside temperature forecast: rising from 18 °C to 34 °C by noon.",
            "NOTE: zone_ai temperature sensor has reduced confidence — "
            "cross-reference with load and cooling physics.",
        ],
    )

    # Attach the per-step outside-temp and wet-bulb curves so environment.py
    # can update outside_temp_c each step if it chooses to.
    # They are stored as facility attributes; environment.py may read them
    # via facility._outside_temp_curve[step] if it wants diurnal weather.
    # This is optional — the scenario is valid without it; outside_temp_c
    # defaults to the initial value and advance_load() handles IT load.
    facility._outside_temp_curve = outside_temps
    facility._wet_bulb_curve = wet_bulbs

    return facility