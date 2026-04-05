"""
Hard scenario: "Cascading Failure + Carbon-Aware Triage"

Four zones:
  zone_ai_1    — critical priority, 500 kW baseline  (must not exceed 30 °C for 5+ steps)
  zone_ai_2    — critical priority, 480 kW baseline  (same constraint)
  zone_storage — medium priority,   200 kW baseline
  zone_infra   — low priority,      120 kW baseline

Fault schedule (injected via FacilityState.chiller_fault_step):
  Step 15 → COP begins degrading: 3.5 → 0.8 over 5 steps
  Step 20 → Chiller goes fully offline (chiller_active = False)

After step 20 the agent has ONLY fan + free-air cooling available.
Outside temperature and wet-bulb follow a realistic 24-hour cycle,
giving meaningful free-cooling potential only at night (steps 0–30
and 240–288).

Carbon intensity follows the default 24-hr curve:
  Steps   0–40  : low     (night renewables)
  Steps  80–160 : high→critical_high (midday grid demand)
  Steps 200–288 : medium→low (evening wind)

Hard termination: any critical zone > 32 °C for 5+ consecutive steps → score = 0.

Episode: 288 steps × 5 min = 24 hours simulated time.
Start time: 00:00 — full diurnal cycle.
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

_START_HOUR = 0.0       # 00:00 — full 24-hr episode
_PID_BASELINE_PUE = 1.48   # PID fails badly after chiller loss; set low to be fair


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
    zone_ai_1 = ZoneState(
        zone_id="zone_ai_1",
        temp_c=23.0,
        it_load_kw=500.0 * 0.72,       # 72 % of base at 00:00
        fan_speed_pct=70.0,
        cooling_capacity_kw=560.0,
        setpoint_c=20.0,
        humidity_pct=44.0,
        sensor_faulty=False,
        hot_aisle_temp_c=31.0,
        supply_air_temp_c=20.0,
        supply_air_temp_setpoint_c=20.0,
        zone_priority=PRIORITY_CRITICAL,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=500.0,
        it_load_pct=0.72,
    )

    # ── zone_ai_2 — critical, secondary AI inference cluster ──────────────────
    zone_ai_2 = ZoneState(
        zone_id="zone_ai_2",
        temp_c=23.5,
        it_load_kw=480.0 * 0.72,
        fan_speed_pct=68.0,
        cooling_capacity_kw=540.0,
        setpoint_c=20.0,
        humidity_pct=44.0,
        sensor_faulty=False,
        hot_aisle_temp_c=31.5,
        supply_air_temp_c=20.0,
        supply_air_temp_setpoint_c=20.0,
        zone_priority=PRIORITY_CRITICAL,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=480.0,
        it_load_pct=0.72,
    )

    # ── zone_storage — medium priority ────────────────────────────────────────
    # Warmer at start — agent may need to sacrifice this zone for the AI clusters.
    zone_storage = ZoneState(
        zone_id="zone_storage",
        temp_c=24.0,
        it_load_kw=200.0 * 0.72,
        fan_speed_pct=55.0,
        cooling_capacity_kw=230.0,
        setpoint_c=22.0,
        humidity_pct=47.0,
        sensor_faulty=False,
        hot_aisle_temp_c=32.0,
        supply_air_temp_c=22.0,
        supply_air_temp_setpoint_c=22.0,
        zone_priority=PRIORITY_MEDIUM,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=200.0,
        it_load_pct=0.72,
    )

    # ── zone_infra — low priority, monitoring / logging ───────────────────────
    # Cheapest to sacrifice. Higher starting temp to create immediate tension.
    zone_infra = ZoneState(
        zone_id="zone_infra",
        temp_c=25.0,
        it_load_kw=120.0 * 0.72,
        fan_speed_pct=50.0,
        cooling_capacity_kw=140.0,
        setpoint_c=23.0,
        humidity_pct=48.0,
        sensor_faulty=False,
        hot_aisle_temp_c=33.0,
        supply_air_temp_c=23.0,
        supply_air_temp_setpoint_c=23.0,
        zone_priority=PRIORITY_LOW,
        sensor_drift_c=0.0,
        sensor_confidence=1.0,
        base_it_load_kw=120.0,
        it_load_pct=0.72,
    )

    outside_temps = _hard_outside_temp_curve()
    wet_bulbs = _hard_wet_bulb_curve()

    facility = FacilityState(
        zones=[zone_ai_1, zone_ai_2, zone_storage, zone_infra],
        outside_temp_c=outside_temps[0],       # 16 °C at midnight
        # chiller — healthy at start; fault injected at step 15
        chiller_active=True,
        chiller_cop=3.5,
        chiller_setpoint_c=10.0,
        chiller_fault_step=15,                 # ← fault trigger
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
        grid_carbon_intensity_normalized=0.20,  # 00:00 = 0.20 per default curve
        # load — heavy inference load throughout
        load_curve=_hard_load_curve(),
        # grader reference
        pid_baseline_pue=_PID_BASELINE_PUE,
        # context — rich text for LLM agents
        maintenance_notes=[],
        upcoming_events=[
            "CRITICAL: Chiller health monitoring has flagged an anomaly. "
            "COP degradation may begin within the next 75 minutes.",
            "Carbon intensity forecast: LOW until 06:00, rising to CRITICAL_HIGH "
            "between 10:00–16:00, easing to MEDIUM by 20:00.",
            "zone_ai_1 and zone_ai_2 are running active inference workloads. "
            "SLA requires cold-aisle temp < 30 °C at all times.",
            "zone_infra can tolerate up to 29 °C for short periods without data loss.",
            "Free-cooling via economiser is available while outside temp < 18 °C "
            "(approximately the first 2.5 hours and the final hour of the episode).",
        ],
    )

    # Attach per-step weather curves for environment.py to use if desired
    facility._outside_temp_curve = outside_temps
    facility._wet_bulb_curve = wet_bulbs

    return facility