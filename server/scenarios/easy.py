"""
Easy scenario: "Thermal Runaway Recovery"

Single zone. Zone starts at 28.5 C (already overheating).
Steady IT load of 450 kW. Outside temp 32 C (hot summer day).
Grid: medium carbon. No faults. Full cooling capacity available.

Agent must bring the zone back into [18-27 C], then maintain it
efficiently -- not just pin fans at 100 % forever.

Episode: 20 steps x 12 min/step = 4 hours of simulated time (condensed, step_scale=2.4).
Thermal physics run at 5-min granularity; clock and load advance at 12-min/step.
Start time: 14:00 (peak carbon / peak outside temperature).
"""

from ..simulation import (
    FacilityState,
    ZoneState,
    PRIORITY_MEDIUM,
    _default_carbon_curve,
    _default_load_curve,
)


# -- Scenario constants --------------------------------------------------------

_START_HOUR = 14.0          # 14:00 -- peak load / carbon window
_OUTSIDE_TEMP_C = 32.0      # hot summer day
_WET_BULB_C = 22.0          # moderate humidity, minimal free-cooling headroom

_ZONE_ID = "zone_main"
_IT_LOAD_KW = 450.0
_START_TEMP_C = 28.5        # already outside safe band [18-27]
_FAN_SPEED_PCT = 60.0       # mid-range -- not already maxed out
_COOLING_CAPACITY_KW = 480.0
_SUPPLY_SETPOINT_C = 22.0   # default setpoint
_HUMIDITY_PCT = 52.0

# PID baseline PUE pre-computed for this scenario (fixed outside temp + load)
# Derived by running pid_baseline.py on the easy scenario for 48 steps.
_PID_BASELINE_PUE = 1.58


def build_easy_scenario(seed: int = 0) -> FacilityState:
    """
    Construct and return the initial FacilityState for the easy task.

    The seed parameter is accepted for API compatibility but this scenario
    is deterministic -- load and outside temp do not vary with seed.
    Randomness enters only through sensor noise in ZoneState.reported_temp_c.
    """
    zone = ZoneState(
        zone_id=_ZONE_ID,
        temp_c=_START_TEMP_C,
        it_load_kw=_IT_LOAD_KW,
        fan_speed_pct=_FAN_SPEED_PCT,
        cooling_capacity_kw=_COOLING_CAPACITY_KW,
        # V2 fields
        setpoint_c=_SUPPLY_SETPOINT_C,
        humidity_pct=_HUMIDITY_PCT,
        sensor_faulty=False,
        hot_aisle_temp_c=_START_TEMP_C + 8.0,   # __post_init__ would set this anyway
        supply_air_temp_c=_SUPPLY_SETPOINT_C,
        supply_air_temp_setpoint_c=_SUPPLY_SETPOINT_C,
        zone_priority=PRIORITY_MEDIUM,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=_IT_LOAD_KW,
        it_load_pct=1.0,
        thermal_mass_kj_per_k=850.0,   # reference zone; matches global default
    )

    # Easy scenario uses a flat load curve -- no diurnal variation.
    # All 24 buckets set to 1.0 so advance_load() leaves IT load unchanged.
    flat_load_curve = [1.0] * 24

    facility = FacilityState(
        zones=[zone],
        outside_temp_c=_OUTSIDE_TEMP_C,
        # chiller
        chiller_active=True,
        chiller_cop=3.5,
        chiller_setpoint_c=10.0,
        chiller_fault_step=-1,      # no fault
        chiller_fault_level=0.0,
        # weather
        wet_bulb_temp_c=_WET_BULB_C,
        # time
        timestamp_hour=_START_HOUR,
        step_number=0,
        # power
        ups_efficiency=0.96,
        # carbon -- medium band at 14:00 per default curve (0.87); use default curve
        grid_carbon_curve=_default_carbon_curve(),
        grid_carbon_intensity="medium",
        grid_carbon_intensity_normalized=0.87,
        # load -- flat (steady state)
        load_curve=flat_load_curve,
        # grader reference
        pid_baseline_pue=_PID_BASELINE_PUE,
        # context
        maintenance_notes=[],
        upcoming_events=[
            "Steady load expected for the next 4 hours.",
            "Outside temperature forecast: 32 C sustained.",
        ],
    )

    return facility