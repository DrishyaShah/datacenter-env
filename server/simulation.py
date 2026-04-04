"""
Data Centre Thermal Simulation
Physics-based single-zone server room model.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, List

THERMAL_MASS_KJ_PER_K = 850.0
SECONDS_PER_STEP = 300  # 5-minute steps
IT_HEAT_FRACTION = 0.98


@dataclass
class ZoneState:
    zone_id: str
    temp_c: float
    it_load_kw: float
    fan_speed_pct: float
    cooling_capacity_kw: float
    setpoint_c: float = 22.0
    humidity_pct: float = 45.0
    sensor_faulty: bool = False

    @property
    def reported_temp_c(self) -> float:
        if self.sensor_faulty:
            return round(self.temp_c + 9.0 + random.gauss(0, 0.5), 2)
        return round(self.temp_c + random.gauss(0, 0.1), 2)

    @property
    def actual_cooling_kw(self) -> float:
        return self.cooling_capacity_kw * (self.fan_speed_pct / 100.0)

    @property
    def fan_power_kw(self) -> float:
        return 8.0 * (self.fan_speed_pct / 100.0) ** 3

    def step_thermal(self, outside_temp_c: float):
        # All power in kW = kJ/s
        heat_in = self.it_load_kw * IT_HEAT_FRACTION   # kJ/s
        heat_out = self.actual_cooling_kw               # kJ/s
        envelope = 0.5 * (self.temp_c - outside_temp_c)  # kJ/s
        net_kw = heat_in - heat_out - envelope          # kJ/s
        net_kj = net_kw * SECONDS_PER_STEP             # kJ over the step
        self.temp_c = round(self.temp_c + net_kj / THERMAL_MASS_KJ_PER_K, 3)
        if self.temp_c > 26:
            self.humidity_pct = min(70, self.humidity_pct + 0.5)
        elif self.temp_c < 20:
            self.humidity_pct = max(30, self.humidity_pct - 0.3)


@dataclass
class FacilityState:
    zones: List[ZoneState]
    outside_temp_c: float
    chiller_active: bool = True
    chiller_cop: float = 3.5
    ups_efficiency: float = 0.96
    step_number: int = 0
    timestamp_hour: float = 14.0
    grid_carbon_intensity: str = "medium"
    maintenance_notes: List[str] = field(default_factory=list)
    upcoming_events: List[str] = field(default_factory=list)

    @property
    def total_it_load_kw(self) -> float:
        return sum(z.it_load_kw for z in self.zones)

    @property
    def total_fan_power_kw(self) -> float:
        return sum(z.fan_power_kw for z in self.zones)

    @property
    def chiller_power_kw(self) -> float:
        if not self.chiller_active:
            return 0.0
        return sum(z.actual_cooling_kw for z in self.zones) / self.chiller_cop

    @property
    def pue(self) -> float:
        it = self.total_it_load_kw / self.ups_efficiency
        cooling = self.total_fan_power_kw + self.chiller_power_kw
        return round((it + cooling) / max(self.total_it_load_kw, 1), 4)

    def advance_time(self):
        self.timestamp_hour = (self.timestamp_hour + SECONDS_PER_STEP / 3600) % 24
        self.step_number += 1

    def to_observation_dict(self) -> dict:
        return {
            "step": self.step_number,
            "timestamp_hour": round(self.timestamp_hour, 2),
            "outside_temp_c": round(self.outside_temp_c, 1),
            "chiller_active": self.chiller_active,
            "chiller_cop": self.chiller_cop,
            "grid_carbon_intensity": self.grid_carbon_intensity,
            "current_pue": self.pue,
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "reported_temp_c": z.reported_temp_c,
                    "it_load_kw": round(z.it_load_kw, 1),
                    "fan_speed_pct": round(z.fan_speed_pct, 1),
                    "cooling_capacity_kw": z.cooling_capacity_kw,
                    "humidity_pct": round(z.humidity_pct, 1),
                    "setpoint_c": z.setpoint_c,
                }
                for z in self.zones
            ],
            "maintenance_notes": self.maintenance_notes,
            "upcoming_events": self.upcoming_events,
        }


def build_easy_scenario(seed: Optional[int] = None) -> FacilityState:
    """Task 1: Single zone, steady load, agent must cool down an overheating room."""
    if seed is not None:
        random.seed(seed)
    zone = ZoneState(
        zone_id="zone_1",
        temp_c=28.5,
        it_load_kw=450.0,
        fan_speed_pct=60.0,
        cooling_capacity_kw=600.0,
        setpoint_c=22.0,
        humidity_pct=52.0,
    )
    return FacilityState(
        zones=[zone],
        outside_temp_c=32.0 + random.uniform(-2, 2),
        chiller_active=True,
        grid_carbon_intensity="medium",
        maintenance_notes=[],
        upcoming_events=["Routine maintenance window at 23:00"],
        timestamp_hour=14.0,
    )