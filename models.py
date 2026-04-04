# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv typed models: Observation, Action, Reward
"""

from openenv.core.env_server.types import Action, Observation
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# class DatacenterAction(Action):
#     """Action for the Datacenter Env environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class DatacenterObservation(Observation):
#     """Observation from the Datacenter Env environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")

# ── Observation ───────────────────────────────────────────────────────────────
 
class ZoneObservation(BaseModel):
    zone_id: str
    reported_temp_c: float = Field(..., description="Reported air temperature in Celsius (may be faulty)")
    it_load_kw: float = Field(..., description="Current IT equipment power draw (kW)")
    fan_speed_pct: float = Field(..., description="Supply fan speed percentage (0-100)")
    cooling_capacity_kw: float = Field(..., description="Max cooling capacity at full fan speed (kW)")
    humidity_pct: float = Field(..., description="Relative humidity percentage")
    setpoint_c: float = Field(..., description="Target supply air temperature (°C)")
 
 
class DCObservation(Observation):
    step: int
    timestamp_hour: float = Field(..., description="Hour of day (0-24)")
    outside_temp_c: float = Field(..., description="Outdoor air temperature (°C)")
    chiller_active: bool
    chiller_cop: float = Field(..., description="Chiller coefficient of performance")
    grid_carbon_intensity: str = Field(..., description="low | medium | high | critical_high")
    current_pue: float = Field(..., description="Power Usage Effectiveness (1.0 = perfect)")
    zones: List[ZoneObservation]
    maintenance_notes: List[str] = Field(default_factory=list)
    upcoming_events: List[str] = Field(default_factory=list)
 
 
# ── Action ────────────────────────────────────────────────────────────────────
 
class ZoneAdjustment(BaseModel):
    zone_id: str
    fan_speed_pct: Optional[float] = Field(None, ge=0, le=100, description="New fan speed (0-100), omit to keep unchanged")
 
 
class DCAction(Action):
    """
    Agent action for one environment step.
    The agent adjusts fan speeds per zone.
    All fields are optional — omitting a zone keeps its current settings.
    """
    zone_adjustments: List[ZoneAdjustment] = Field(
        default_factory=list,
        description="List of per-zone fan speed adjustments"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's explanation of why it took this action (used for grading hard task)"
    )
 
 
# ── Reward ────────────────────────────────────────────────────────────────────
 
class DCReward(BaseModel):
    total: float = Field(..., description="Combined reward for this step (bounded -1 to +1)")
    pue_reward: float = Field(..., description="Reward component for PUE improvement")
    temperature_penalty: float = Field(..., description="Penalty for zones outside safe range (<=0)")
    humidity_penalty: float = Field(..., description="Penalty for humidity violations (<=0)")
    breakdown: Dict[str, float] = Field(default_factory=dict)
 
 
# ── Step Result ───────────────────────────────────────────────────────────────
 
class StepResult(BaseModel):
    observation: DCObservation
    reward: float
    reward_detail: DCReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
 
 
class ResetResult(BaseModel):
    observation: DCObservation
    info: Dict[str, Any] = Field(default_factory=dict)
 