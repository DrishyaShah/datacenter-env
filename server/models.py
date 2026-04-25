"""
OpenEnv typed models: Observation, Action, Reward -- V2
"""

from openenv.core.env_server.types import Action, Observation
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -- Zone Observation ----------------------------------------------------------

class ZoneObservation(BaseModel):
    zone_id: str

    # -- Thermal readings ------------------------------------------------------
    cold_aisle_temp_c: float = Field(
        ...,
        description=(
            "Primary cold-aisle temperature sensor reading (C). "
            "For zones with sensor_confidence < 0.7 this value may be drifted "
            "-- it reflects the same faulty sensor as reported_temp_c. "
            "Use hot_aisle_temp_c, supply_air_temp_c, and it_load_kw to "
            "cross-check the true thermal state when sensor_confidence is low."
        ),
    )
    hot_aisle_temp_c: float = Field(
        ..., description="Return-air temperature from server exhausts (C). Always accurate."
    )
    reported_temp_c: float = Field(
        ...,
        description=(
            "Secondary sensor reading (C). Typically matches cold_aisle_temp_c; "
            "cross-checking both values plus hot_aisle physics can reveal sensor drift."
        ),
    )
    supply_air_temp_c: float = Field(
        ..., description="Actual delivered supply air temperature (C)"
    )
    supply_air_temp_setpoint_c: float = Field(
        ..., description="Agent-controlled supply air temperature setpoint [16-26] (C)"
    )

    # -- Load ------------------------------------------------------------------
    it_load_kw: float = Field(..., description="Current IT equipment power draw (kW)")
    it_load_pct: float = Field(
        ..., description="Normalised IT load relative to zone baseline [0-1]"
    )

    # -- Fan / cooling ---------------------------------------------------------
    fan_speed_pct: float = Field(..., description="Current fan speed (0-100 %)")
    cooling_capacity_kw: float = Field(
        ..., description="Max cooling capacity at full fan speed (kW)"
    )

    # -- Environment -----------------------------------------------------------
    humidity_pct: float = Field(..., description="Relative humidity (%)")

    # -- Sensor metadata -------------------------------------------------------
    sensor_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Reliability weight of this zone's sensor reading [0.0-1.0]"
    )
    zone_priority: int = Field(
        ..., ge=0, le=2,
        description="Static criticality label: 0=low, 1=medium, 2=critical"
    )

    # -- Forecast --------------------------------------------------------------
    load_forecast_next_hour: float = Field(
        default=0.0,
        description="Predicted IT load in the next 60 minutes (kW)"
    )


# -- Facility Observation ------------------------------------------------------

class DCObservation(Observation):
    step: int

    # -- Time ------------------------------------------------------------------
    timestamp_hour: float = Field(..., description="Hour of day [0-24]")
    timestamp_day_sin: float = Field(
        ..., description="sin(2 x hour/24) -- cyclical time encoding"
    )
    timestamp_day_cos: float = Field(
        ..., description="cos(2 x hour/24) -- cyclical time encoding"
    )

    # -- Weather ---------------------------------------------------------------
    outside_temp_c: float = Field(..., description="Outdoor dry-bulb temperature (C)")
    wet_bulb_temp_c: float = Field(
        ..., description="Outdoor wet-bulb temperature -- determines free-cooling potential (C)"
    )

    # -- Chiller ---------------------------------------------------------------
    chiller_active: bool
    chiller_setpoint_c: float = Field(
        ..., description="Current chilled-water setpoint [6-15] (C)"
    )
    chiller_cop: float = Field(..., description="Chiller coefficient of performance")
    chiller_fault_detected: bool = Field(
        default=False,
        description="Observable anomaly signal -- True when COP has degraded abnormally",
    )
    chiller_fault_status: str = Field(
        default="nominal",
        description=(
            "Chiller health status: "
            "'nominal' = fully operational; "
            "'degrading' = COP dropping, still providing partial cooling; "
            "'offline' = chiller has failed and provides zero cooling. "
            "When 'offline', setting chiller_active=true in your action has no effect."
        ),
    )

    # -- Power -----------------------------------------------------------------
    ups_efficiency: float = Field(..., description="UPS efficiency [0-1]")
    current_pue: float = Field(..., description="Real-time Power Usage Effectiveness (1.0 = perfect)")
    free_cooling_potential: float = Field(
        default=0.0,
        description="Fraction of cooling that could be met by free-air economiser [0-1]"
    )

    # -- Carbon ----------------------------------------------------------------
    grid_carbon_intensity: str = Field(
        ..., description="Human-readable label: low | medium | high | critical_high"
    )
    carbon_intensity_normalized: float = Field(
        ..., ge=0.0, le=1.0,
        description="Numeric carbon intensity [0.0-1.0] for reward computation"
    )

    # -- Load phase ------------------------------------------------------------
    load_curve_phase: str = Field(
        default="idle",
        description="Current phase of the diurnal load curve: ramp_up | peak | ramp_down | idle"
    )

    # -- Zones -----------------------------------------------------------------
    zones: List[ZoneObservation]

    # -- Temporal history buffer -----------------------------------------------
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Last 3 step snapshots. Each entry: "
            "{cold_aisle_temp, hot_aisle_temp, fan_speed_pct, pue} per zone, "
            "flattened as history_t-1, history_t-2, history_t-3"
        )
    )

    # -- Event / context block -------------------------------------------------
    sla_violation_streak: int = Field(
        default=0,
        description="Consecutive steps in which any zone was outside the safe temperature band"
    )
    maintenance_active: bool = Field(
        default=False,
        description="True if any zone is currently in a maintenance window"
    )
    maintenance_notes: List[str] = Field(default_factory=list)
    upcoming_events: List[str] = Field(default_factory=list)

    # -- Active alerts (environment-computed, not agent-injected) --------------
    active_alerts: List[str] = Field(
        default_factory=list,
        description=(
            "Structured alert strings generated by the environment at each step. "
            "Covers: chiller fault/offline, zone overheating/overcooling, "
            "sensor faults, temperature trends, efficiency hints, SLA streaks. "
            "Agents should act on these before inspecting raw numeric fields."
        ),
    )


# -- Action --------------------------------------------------------------------

class ZoneAdjustment(BaseModel):
    zone_id: str
    fan_speed_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Target fan speed for this zone (0-100 %)"
    )
    supply_air_temp_setpoint_c: float = Field(
        ..., ge=16.0, le=26.0,
        description="Target supply air temperature setpoint for this zone [16-26] (C)"
    )


class DCAction(Action):
    """
    Agent action for one environment step.

    Three control levers per zone plus one facility-level lever.
    All zone adjustments must be submitted together; omitting a zone
    leaves its current settings unchanged.
    """

    zone_adjustments: List[ZoneAdjustment] = Field(
        default_factory=list,
        description="Per-zone fan speed and supply air temperature setpoint adjustments"
    )

    # -- Facility-level levers -------------------------------------------------
    chiller_setpoint_c: float = Field(
        default=10.0, ge=6.0, le=15.0,
        description="Facility-wide chilled-water setpoint [6-15] (C)"
    )
    chiller_active: bool = Field(
        default=True,
        description=(
            "Toggle chiller on/off. Turning off saves significant power but "
            "temperatures will rise for 3-5 steps before consequences appear."
        )
    )

    # -- Reasoning (graded in hard task) --------------------------------------
    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Agent's explanation of its decision. "
            "Graded in the hard task -- coherent crisis reasoning is rewarded; "
            "inconsistency between stated reasoning and actual action is penalised."
        )
    )


# -- Reward breakdown ----------------------------------------------------------

class DCReward(BaseModel):
    total: float = Field(..., description="Combined reward for this step")

    # -- Component rewards -----------------------------------------------------
    temp_reward: float = Field(
        default=0.0,
        description="R_temp -- temperature compliance reward (priority-weighted)"
    )
    pue_reward: float = Field(
        default=0.0,
        description="R_pue -- efficiency reward relative to PID baseline"
    )
    carbon_reward: float = Field(
        default=0.0,
        description="R_carbon -- penalty for high cooling power during dirty-grid periods (0)"
    )

    # -- Penalties -------------------------------------------------------------
    safety_penalty: float = Field(
        default=0.0,
        description="P_safety -- hard penalty for temperature safety violations (0)"
    )
    roughness_penalty: float = Field(
        default=0.0,
        description="P_roughness -- penalty for abrupt action changes (0)"
    )

    # -- Bonus -----------------------------------------------------------------
    stability_bonus: float = Field(
        default=0.0,
        description="B_stability -- compounding bonus for sustained temperature compliance (0)"
    )

    # -- Legacy / extra --------------------------------------------------------
    # Kept for backward compatibility with any existing grader consumers
    temperature_penalty: float = Field(
        default=0.0,
        description="Alias for safety_penalty (legacy field -- prefer safety_penalty)"
    )
    humidity_penalty: float = Field(
        default=0.0,
        description="Penalty for humidity violations (0)"
    )

    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-zone or per-component reward detail for debugging"
    )


# -- Step / reset results ------------------------------------------------------

class StepResult(BaseModel):
    observation: DCObservation
    reward: float
    reward_detail: DCReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: DCObservation
    info: Dict[str, Any] = Field(default_factory=dict)