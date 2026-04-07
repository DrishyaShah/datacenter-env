"""
Data Centre Thermal Simulation — V2
Physics-based multi-zone server room model.

Responsibilities (pure physics only):
  - ZoneState  — per-zone thermal + sensor state
  - FacilityState — facility-level state including chiller, carbon, load curves
  - step_thermal(), apply_action_with_rate_limiting(), advance_time()
  - Sensor drift, chiller fault injection, free-cooling, load/carbon advancement

No RL concepts. No reward logic. Just physics.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ── Physical constants ────────────────────────────────────────────────────────

THERMAL_MASS_KJ_PER_K = 850.0
SECONDS_PER_STEP = 300          # 5-minute steps
IT_HEAT_FRACTION = 0.98

# Action rate limits (enforced by apply_action_with_rate_limiting)
MAX_FAN_DELTA_PER_STEP = 20.0          # ± % per step
MAX_SUPPLY_TEMP_DELTA = 2.0            # ± °C per step
MAX_CHILLER_SETPOINT_DELTA = 1.0       # ± °C per step

# Chiller operating bounds
CHILLER_SETPOINT_MIN = 6.0
CHILLER_SETPOINT_MAX = 15.0

# Fan / supply air bounds
FAN_SPEED_MIN = 0.0
FAN_SPEED_MAX = 100.0
SUPPLY_AIR_TEMP_MIN = 16.0
SUPPLY_AIR_TEMP_MAX = 26.0

# Reference delta-T used in cooling-power scaling (°C).
# Must match the natural hot-aisle rise at design conditions:
#   heat_in / (max_mass_flow × cp_air) = 441 / (50 × 1.006) ≈ 8.77 °C
# At REF=15 effective cooling at equilibrium was only 280 kW < 441 kW heat-in,
# making the zone impossible to cool at any fan speed.  REF=9 gives 468 kW.
COOLING_DELTA_T_REF = 9.0

# Mass-flow reference: easy zone (480 kW capacity) at 100% fan moves 50 kg/s.
# All other zones scale proportionally so every zone has the same 6% cooling
# headroom above heat-in: capacity / (mass_flow_max × 1.006 × REF) ≈ 1.06.
MASS_FLOW_REF_CAPACITY_KW = 480.0   # easy zone cooling capacity (calibration reference)
MASS_FLOW_REF_KGS         = 50.0    # mass flow at 100% fan for the reference zone

# Cube-law fan power coefficient (kW at 100 %)
FAN_POWER_MAX_KW = 8.0

# Envelope heat-transfer coefficient (kW / °C)
ENVELOPE_CONDUCTANCE = 0.5

# Hot-aisle temperature rise coefficient
HOT_AISLE_RISE_COEFF = 5.0


# ── Zone priority labels ──────────────────────────────────────────────────────

PRIORITY_LOW = 0
PRIORITY_MEDIUM = 1
PRIORITY_CRITICAL = 2


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ZoneState:
    """Per-zone thermal and sensor state (V2 — extended from V1)."""

    # ── Identity ──────────────────────────────────────────────────────────────
    zone_id: str

    # ── Thermal state ─────────────────────────────────────────────────────────
    temp_c: float                           # cold-aisle temperature (true value)
    it_load_kw: float
    fan_speed_pct: float
    cooling_capacity_kw: float

    # ── V1 fields (kept) ──────────────────────────────────────────────────────
    setpoint_c: float = 22.0               # legacy alias; use supply_air_temp_setpoint_c
    humidity_pct: float = 45.0
    sensor_faulty: bool = False

    # ── V2 new fields ─────────────────────────────────────────────────────────
    hot_aisle_temp_c: float = 0.0          # return-air temperature (computed)
    supply_air_temp_c: float = 18.0        # actual delivered supply air temp
    supply_air_temp_setpoint_c: float = 22.0  # agent-controlled setpoint
    zone_priority: int = PRIORITY_MEDIUM   # 0=low, 1=medium, 2=critical
    sensor_drift_c: float = 0.0            # cumulative sensor drift (°C)
    sensor_confidence: float = 1.0        # [0.0–1.0] reliability weight
    base_it_load_kw: float = 0.0          # baseline load before diurnal variation
    it_load_pct: float = 0.0              # normalised load [0–1]
    thermal_mass_kj_per_k: float = 850.0  # room thermal mass (kJ/K); scale per zone size

    def __post_init__(self):
        if self.base_it_load_kw == 0.0:
            self.base_it_load_kw = self.it_load_kw
        if self.hot_aisle_temp_c == 0.0:
            self.hot_aisle_temp_c = self.temp_c + 8.0
        self.supply_air_temp_c = self.supply_air_temp_setpoint_c

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def reported_temp_c(self) -> float:
        """Sensor reading — may be drifted if sensor_faulty is True."""
        noise = random.gauss(0, 0.1)
        if self.sensor_faulty:
            return round(self.temp_c + self.sensor_drift_c + noise, 2)
        return round(self.temp_c + noise, 2)

    @property
    def actual_cooling_kw(self) -> float:
        """Cooling power delivered (before delta-T scaling)."""
        return self.cooling_capacity_kw * (self.fan_speed_pct / 100.0)

    @property
    def fan_power_kw(self) -> float:
        """Fan electrical power (cube law), scaled by zone cooling capacity."""
        capacity_ratio = self.cooling_capacity_kw / MASS_FLOW_REF_CAPACITY_KW
        return FAN_POWER_MAX_KW * capacity_ratio * (self.fan_speed_pct / 100.0) ** 3

    # ── Thermal step ──────────────────────────────────────────────────────────

    def step_thermal(self, outside_temp_c: float, supply_air_temp_c: Optional[float] = None):
        """
        Advance zone temperature by one time step.

        V2 upgrade: cooling power is scaled by the delta-T between the hot aisle
        and the delivered supply air temperature (not just fan speed alone).
        """
        if supply_air_temp_c is None:
            supply_air_temp_c = self.supply_air_temp_c

        heat_in_kw = self.it_load_kw * IT_HEAT_FRACTION

        # Effective cooling scales with temperature driving force
        delta_t = max(self.hot_aisle_temp_c - supply_air_temp_c, 0.0)
        scaling = delta_t / COOLING_DELTA_T_REF
        effective_cooling_kw = self.actual_cooling_kw * scaling
        effective_cooling_kw = min(effective_cooling_kw, heat_in_kw * 3.0)
        # Envelope loss / gain (positive = heat flowing out when room is warm)
        envelope_kw = ENVELOPE_CONDUCTANCE * (self.temp_c - outside_temp_c)

        net_kw = heat_in_kw - effective_cooling_kw - envelope_kw
        net_kj = net_kw * SECONDS_PER_STEP
        delta_temp = net_kj / self.thermal_mass_kj_per_k   # per-zone thermal mass
        # delta_temp = max(-5.0, min(5.0, delta_temp))
        delta_temp = max(-2.0, min(2.0, delta_temp))
        self.temp_c = round(self.temp_c + delta_temp, 3)

        # Derived temperatures
        # denominator = max(effective_cooling_kw, 0.1)
        # self.hot_aisle_temp_c = round(
        #     self.temp_c + (heat_in_kw / denominator) * HOT_AISLE_RISE_COEFF, 3
        # )
        # self.hot_aisle_temp_c = max(self.hot_aisle_temp_c, self.temp_c + 1.0)

        # Mass flow scales with zone cooling capacity so every zone has the
        # same ~6% cooling headroom above heat-in (fixes small-zone uncoolability).
        mass_flow = (
            (self.cooling_capacity_kw / MASS_FLOW_REF_CAPACITY_KW)
            * MASS_FLOW_REF_KGS
            * (self.fan_speed_pct / 100.0)
        )  # kg/s
        if self.fan_speed_pct > 0.5:
            self.hot_aisle_temp_c = round(
                supply_air_temp_c + heat_in_kw / (mass_flow * 1.006), 3
            )
        else:
            self.hot_aisle_temp_c = round(min(supply_air_temp_c + 50.0, 85.0), 3)
        self.supply_air_temp_c = supply_air_temp_c

        # Cold-aisle temperature floor: the zone cannot be colder than the
        # supply air being delivered into it (physically impossible).
        self.temp_c = round(max(self.temp_c, supply_air_temp_c), 3)

        # Humidity heuristic
        if self.temp_c > 26:
            self.humidity_pct = min(70.0, self.humidity_pct + 0.5)
        elif self.temp_c < 20:
            self.humidity_pct = max(30.0, self.humidity_pct - 0.3)

        # Normalised load
        peak_load = max(self.base_it_load_kw, 1.0)
        self.it_load_pct = round(min(self.it_load_kw / peak_load, 1.0), 4)


# ── Facility state ────────────────────────────────────────────────────────────

@dataclass
class FacilityState:
    """Facility-level state (V2 — extended from V1)."""

    # ── Core ──────────────────────────────────────────────────────────────────
    zones: List[ZoneState]
    outside_temp_c: float

    # ── V1 fields (kept) ──────────────────────────────────────────────────────
    chiller_active: bool = True
    chiller_cop: float = 3.5
    ups_efficiency: float = 0.96
    step_number: int = 0
    timestamp_hour: float = 14.0
    grid_carbon_intensity: str = "medium"
    maintenance_notes: List[str] = field(default_factory=list)
    upcoming_events: List[str] = field(default_factory=list)

    # ── V2 new fields ─────────────────────────────────────────────────────────
    chiller_setpoint_c: float = 10.0           # agent-controlled, [6–15]°C
    chiller_fault_level: float = 0.0           # 0 = healthy, 1 = full failure
    chiller_fault_step: int = -1               # step when fault triggers (-1 = never)
    wet_bulb_temp_c: float = 18.0              # enables free-cooling logic
    grid_carbon_curve: List[float] = field(default_factory=list)   # 24-hr normalised [0–1]
    load_curve: List[float] = field(default_factory=list)           # 24-hr normalised [0–1]
    pid_baseline_pue: float = 1.55             # pre-computed PID reference PUE
    grid_carbon_intensity_normalized: float = 0.5  # [0–1] numeric companion
    minutes_per_step: float = 5.0                  # sim minutes per env step; set by environment.py for timeline condensation

    # ── Convenience constants ─────────────────────────────────────────────────
    _BASE_CHILLER_COP: float = field(default=3.5, init=False, repr=False)

    def __post_init__(self):
        self._BASE_CHILLER_COP = self.chiller_cop
        if not self.grid_carbon_curve:
            self.grid_carbon_curve = _default_carbon_curve()
        if not self.load_curve:
            self.load_curve = _default_load_curve()

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def total_it_load_kw(self) -> float:
        return sum(z.it_load_kw for z in self.zones)

    @property
    def total_fan_power_kw(self) -> float:
        return sum(z.fan_power_kw for z in self.zones)

    @property
    def effective_chiller_cop(self) -> float:
        """
        COP adjusted for supply-water temperature and outdoor conditions.

        Real-world behaviour:
          - Higher leaving-water temp (chiller_setpoint_c) → less compression work → higher COP.
          - Higher outdoor temp → harder heat rejection → lower COP.
        Fault degradation overrides temperature adjustment: once inject_chiller_fault()
        has modified self.chiller_cop, that modified value is used as-is.
        """
        if self.chiller_fault_level > 0:
            return self.chiller_cop   # fault path; COP already degraded
        cop = self._BASE_CHILLER_COP
        cop *= (1.0 + 0.03 * (self.chiller_setpoint_c - 10.0))   # +3 % per °C higher setpoint
        cop *= (1.0 - 0.02 * max(0.0, self.outside_temp_c - 20.0))  # −2 % per °C outdoor > 20
        return max(1.0, min(6.0, cop))

    @property
    def chiller_power_kw(self) -> float:
        if not self.chiller_active:
            return 0.0
        total_cooling = sum(z.actual_cooling_kw for z in self.zones)
        cop = max(self.effective_chiller_cop, 0.01)
        return total_cooling / cop

    @property
    def pue(self) -> float:
        it = self.total_it_load_kw / max(self.ups_efficiency, 0.01)
        cooling = self.total_fan_power_kw + self.chiller_power_kw
        return round((it + cooling) / max(self.total_it_load_kw, 1.0), 4)

    # ── Time advancement ──────────────────────────────────────────────────────

    def advance_time(self):
        """Tick clock forward by one step (minutes_per_step minutes)."""
        self.timestamp_hour = (self.timestamp_hour + self.minutes_per_step / 60.0) % 24.0
        self.step_number += 1

    # ── Load advancement ──────────────────────────────────────────────────────

    def advance_load(self):
        """
        Update IT load per zone based on the diurnal curve plus random
        Poisson batch-job arrivals (±5 % burst).
        """
        if not self.load_curve:
            return

        hour_idx = int(self.timestamp_hour) % 24
        normalised_load = self.load_curve[hour_idx]

        for zone in self.zones:
            # Scale load around the zone's base
            batch_burst = 0.0
            if random.random() < 0.05:          # 5 % chance of batch arrival
                batch_burst = random.uniform(0.03, 0.08)

            zone.it_load_kw = round(
                zone.base_it_load_kw * (normalised_load + batch_burst), 2
            )
            peak = max(zone.base_it_load_kw, 1.0)
            zone.it_load_pct = round(min(zone.it_load_kw / peak, 1.0), 4)

    # ── Carbon advancement ────────────────────────────────────────────────────

    def advance_carbon(self):
        """Update grid carbon intensity from the 24-hr curve."""
        if not self.grid_carbon_curve:
            return

        hour_idx = int(self.timestamp_hour) % 24
        intensity = self.grid_carbon_curve[hour_idx]
        self.grid_carbon_intensity_normalized = round(intensity, 4)

        # Also update the human-readable label
        if intensity < 0.25:
            self.grid_carbon_intensity = "low"
        elif intensity < 0.55:
            self.grid_carbon_intensity = "medium"
        elif intensity < 0.80:
            self.grid_carbon_intensity = "high"
        else:
            self.grid_carbon_intensity = "critical_high"

    # ── Sensor drift ──────────────────────────────────────────────────────────

    def apply_sensor_drift(self, step: int):
        """
        Gradually drift faulty zone sensors over time.
        Confidence decreases proportionally as drift grows.

        Uses an effective step scaled by minutes_per_step so that drift
        reaches its maximum at the same simulated time regardless of
        whether the episode is condensed (larger minutes_per_step).
        At 5 min/step the sensor stabilises at ~12 °C around step 50.
        At 24 min/step (medium condensed) it stabilises around step 10.
        """
        for zone in self.zones:
            if not zone.sensor_faulty:
                continue
            # Scale step by how many 5-min periods this step represents
            effective_step = int(step * self.minutes_per_step / 5.0)
            target_drift = min(3.0 + effective_step * 0.18, 12.0)
            zone.sensor_drift_c = round(target_drift, 2)
            # Confidence shrinks from 1.0 → ~0.1 as drift grows from 0 → 12
            zone.sensor_confidence = round(max(0.1, 1.0 - zone.sensor_drift_c / 13.0), 3)

    # ── Chiller fault injection ───────────────────────────────────────────────

    def inject_chiller_fault(self, step: int):
        """
        Degrade chiller COP based on fault progression.

        Fault schedule (hard task):
          step 15 → COP drops from 3.5 → 0.8  (partial failure)
          step 20 → chiller goes fully offline
        """
        if self.chiller_fault_step < 0:
            return   # no fault configured for this scenario

        if step < self.chiller_fault_step:
            return   # fault not yet triggered

        steps_since_fault = step - self.chiller_fault_step
        if steps_since_fault < 5:
            # Ramp COP down over 5 steps: 3.5 → 0.8
            progress = steps_since_fault / 5.0
            degraded_cop = self._BASE_CHILLER_COP + progress * (0.8 - self._BASE_CHILLER_COP)
            self.chiller_cop = round(max(degraded_cop, 0.8), 3)
            self.chiller_fault_level = round(progress, 3)
        else:
            # Full failure
            self.chiller_active = False
            self.chiller_cop = 0.0
            self.chiller_fault_level = 1.0

    # ── Free-cooling potential ────────────────────────────────────────────────

    def compute_free_cooling_potential(self) -> float:
        """
        Estimate how much cooling can come from outside air alone.

        Returns a fraction [0.0–1.0] of the total required cooling that
        could be met via economiser / free-air cooling, based on the
        wet-bulb temperature vs. the average zone cold-aisle temperature.

        Rule of thumb: free cooling is viable when wet-bulb < (zone_temp - 4°C).
        """
        if not self.zones:
            return 0.0

        avg_zone_temp = sum(z.temp_c for z in self.zones) / len(self.zones)
        headroom = avg_zone_temp - self.wet_bulb_temp_c - 4.0

        if headroom <= 0.0:
            return 0.0

        # Scale linearly: full free cooling when headroom ≥ 10 °C
        return round(min(headroom / 10.0, 1.0), 3)

    # ── Chiller setpoint → supply air temperature propagation ─────────────────

    def propagate_chiller_setpoint(self):
        """
        Translate the chiller setpoint into actual supply air temperature
        for each zone, accounting for chiller COP and free-cooling potential.

        When the chiller is offline, supply air approaches outdoor wet-bulb.

        Free-cooling (economiser) blending is only applied when outdoor air
        is actually cooler than what the chiller delivers.  If the wet-bulb
        temperature is above the chilled-supply setpoint — e.g. a hot summer
        day with wet-bulb=22 °C and supply setpoint=18 °C — blending outdoor
        air would *raise* the supply temperature and reduce cooling effectiveness.
        In that case, the chilled supply is used as-is.
        """
        free_cooling = self.compute_free_cooling_potential()
        free_cooling_air_temp = self.wet_bulb_temp_c + 2.0

        for zone in self.zones:
            if self.chiller_active:
                # Chiller can deliver down to chiller_setpoint + duct loss
                # but zone setpoint is the agent's actual control lever
                chiller_floor = self.chiller_setpoint_c + 0.5
                # Agent setpoint is bounded below by what the chiller can deliver
                chilled_supply = max(zone.supply_air_temp_setpoint_c, chiller_floor)
            else:
                chilled_supply = free_cooling_air_temp

            # Only blend outdoor (free-cooling) air when it is colder than the
            # chilled supply — otherwise free cooling would warm, not cool.
            if free_cooling > 0.0 and free_cooling_air_temp < chilled_supply:
                effective_supply = (
                    free_cooling * free_cooling_air_temp
                    + (1.0 - free_cooling) * chilled_supply
                )
            else:
                effective_supply = chilled_supply

            zone.supply_air_temp_c = round(
                max(SUPPLY_AIR_TEMP_MIN, min(effective_supply, SUPPLY_AIR_TEMP_MAX)), 2
            )
    # ── Action application with rate-limiting ─────────────────────────────────

    def apply_action_with_rate_limiting(
        self,
        action: "DCAction",
        last_action: "DCAction",
    ) -> Dict[str, bool]:
        """
        Apply agent action, enforcing per-step delta limits.

        Clips any action component that exceeds its rate limit and returns
        an info dict flagging which levers were clipped.

        Returns
        -------
        info : dict
            {
              "chiller_setpoint_clipped": bool,
              "chiller_toggled": bool,
              "zones": {zone_id: {"fan_clipped": bool, "supply_temp_clipped": bool}}
            }
        """
        info: Dict = {
            "chiller_setpoint_clipped": False,
            "chiller_toggled": False,
            "zones": {},
        }

        # ── Facility-level levers ─────────────────────────────────────────────

        # Chiller setpoint
        raw_delta = action.chiller_setpoint_c - last_action.chiller_setpoint_c
        clipped_delta = _clip(raw_delta, MAX_CHILLER_SETPOINT_DELTA)
        if abs(clipped_delta - raw_delta) > 1e-6:
            info["chiller_setpoint_clipped"] = True
        new_setpoint = last_action.chiller_setpoint_c + clipped_delta
        self.chiller_setpoint_c = round(
            _clip_bounds(new_setpoint, CHILLER_SETPOINT_MIN, CHILLER_SETPOINT_MAX), 2
        )

        # Chiller on/off — allowed freely (nuclear option, but costly)
        if self.chiller_fault_level < 1.0:   # cannot re-enable a fully failed chiller
            self.chiller_active = action.chiller_active
            info["chiller_toggled"] = (action.chiller_active != last_action.chiller_active)

        # ── Per-zone levers ───────────────────────────────────────────────────

        zone_map = {z.zone_id: z for z in self.zones}
        last_zone_map = {za.zone_id: za for za in last_action.zone_adjustments}

        for adj in action.zone_adjustments:
            zone = zone_map.get(adj.zone_id)
            if zone is None:
                continue

            last_adj = last_zone_map.get(adj.zone_id)
            last_fan = last_adj.fan_speed_pct if last_adj else zone.fan_speed_pct
            last_supply = (
                last_adj.supply_air_temp_setpoint_c
                if last_adj
                else zone.supply_air_temp_setpoint_c
            )

            zone_info = {"fan_clipped": False, "supply_temp_clipped": False}

            # Fan speed
            fan_raw_delta = adj.fan_speed_pct - last_fan
            fan_clipped_delta = _clip(fan_raw_delta, MAX_FAN_DELTA_PER_STEP)
            if abs(fan_clipped_delta - fan_raw_delta) > 1e-6:
                zone_info["fan_clipped"] = True
            zone.fan_speed_pct = round(
                _clip_bounds(last_fan + fan_clipped_delta, FAN_SPEED_MIN, FAN_SPEED_MAX), 2
            )

            # Supply air temperature setpoint
            supply_raw_delta = adj.supply_air_temp_setpoint_c - last_supply
            supply_clipped_delta = _clip(supply_raw_delta, MAX_SUPPLY_TEMP_DELTA)
            if abs(supply_clipped_delta - supply_raw_delta) > 1e-6:
                zone_info["supply_temp_clipped"] = True
            zone.supply_air_temp_setpoint_c = round(
                _clip_bounds(
                    last_supply + supply_clipped_delta,
                    SUPPLY_AIR_TEMP_MIN,
                    SUPPLY_AIR_TEMP_MAX,
                ),
                2,
            )

            info["zones"][adj.zone_id] = zone_info

        return info

    # ── Full simulation step ──────────────────────────────────────────────────

    def step(
        self,
        action: "DCAction",
        last_action: "DCAction",
    ) -> Dict:
        """
        Advance simulation by one step.

        Order of operations (per spec §4 Transition Dynamics):
          1. Rate-limit and clip action
          2. Apply supply air temp setpoint (propagate chiller → zones)
          3. Update load and carbon from curves
          4. Apply chiller fault (if scheduled)
          5. Thermal step per zone
          6. Sensor drift
          7. Advance time
        """
        # 1. Apply action (with clipping)
        clip_info = self.apply_action_with_rate_limiting(action, last_action)

        # 2. Propagate chiller setpoint → zone supply air temps
        self.propagate_chiller_setpoint()

        # 3. Update load and carbon
        self.advance_load()
        self.advance_carbon()

        # 4. Chiller fault injection
        self.inject_chiller_fault(self.step_number)

        # 5. Thermal step per zone
        for zone in self.zones:
            zone.step_thermal(self.outside_temp_c, zone.supply_air_temp_c)

        # 6. Sensor drift
        self.apply_sensor_drift(self.step_number)

        # 7. Advance time
        self.advance_time()

        return {"action_clipped": clip_info}

    # ── Observation serialisation ─────────────────────────────────────────────

    def to_observation_dict(self) -> dict:
        """Serialise full V2 observation (scalars + text fields)."""
        hour = self.timestamp_hour
        return {
            "step": self.step_number,
            "timestamp_hour": round(hour, 2),
            "timestamp_day_sin": round(math.sin(2 * math.pi * hour / 24.0), 6),
            "timestamp_day_cos": round(math.cos(2 * math.pi * hour / 24.0), 6),
            "outside_temp_c": round(self.outside_temp_c, 1),
            "wet_bulb_temp_c": round(self.wet_bulb_temp_c, 1),
            "chiller_active": self.chiller_active,
            "chiller_setpoint_c": round(self.chiller_setpoint_c, 2),
            "chiller_cop": round(self.chiller_cop, 3),
            "chiller_fault_level": round(self.chiller_fault_level, 3),
            "ups_efficiency": round(self.ups_efficiency, 4),
            "current_pue": self.pue,
            "grid_carbon_intensity": self.grid_carbon_intensity,
            "grid_carbon_intensity_normalized": self.grid_carbon_intensity_normalized,
            "free_cooling_potential": self.compute_free_cooling_potential(),
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "cold_aisle_temp_c": round(z.temp_c, 3),
                    "hot_aisle_temp_c": round(z.hot_aisle_temp_c, 3),
                    "reported_temp_c": z.reported_temp_c,
                    "it_load_kw": round(z.it_load_kw, 1),
                    "it_load_pct": round(z.it_load_pct, 4),
                    "fan_speed_pct": round(z.fan_speed_pct, 1),
                    "supply_air_temp_c": round(z.supply_air_temp_c, 2),
                    "supply_air_temp_setpoint_c": round(z.supply_air_temp_setpoint_c, 2),
                    "cooling_capacity_kw": z.cooling_capacity_kw,
                    "humidity_pct": round(z.humidity_pct, 1),
                    "sensor_confidence": round(z.sensor_confidence, 3),
                    "sensor_drift_c": round(z.sensor_drift_c, 2),
                    "zone_priority": z.zone_priority,
                }
                for z in self.zones
            ],
            "maintenance_notes": self.maintenance_notes,
            "upcoming_events": self.upcoming_events,
        }


# ── Default curves ────────────────────────────────────────────────────────────

def _default_carbon_curve() -> List[float]:
    """
    Realistic 24-hour grid carbon intensity [0–1].
    Low at night (wind/nuclear), peaks around midday (grid demand).
    Indices correspond to hour-of-day (0–23).
    """
    return [
        0.20, 0.18, 0.17, 0.16, 0.17, 0.20,   # 00–05  (night, renewables)
        0.30, 0.45, 0.60, 0.72, 0.80, 0.85,   # 06–11  (morning ramp)
        0.88, 0.90, 0.87, 0.82, 0.75, 0.65,   # 12–17  (midday peak)
        0.55, 0.45, 0.38, 0.32, 0.27, 0.22,   # 18–23  (evening wind)
    ]


def _default_load_curve() -> List[float]:
    """
    Normalised IT load [0–1] across 24 hours.
    Low at night, rises with business hours, peaks around 10–14h.
    """
    return [
        0.55, 0.52, 0.50, 0.50, 0.52, 0.58,   # 00–05  (overnight batch)
        0.65, 0.75, 0.88, 0.95, 0.98, 0.99,   # 06–11  (morning surge)
        1.00, 0.98, 0.96, 0.94, 0.90, 0.85,   # 12–17  (business hours)
        0.78, 0.72, 0.68, 0.64, 0.60, 0.57,   # 18–23  (evening taper)
    ]


# ── Action stubs (imported by modules that call apply_action_with_rate_limiting)
# Full Pydantic models live in models.py; these stubs are used here only for
# type-hint purposes and to keep simulation.py import-free of pydantic.

class _ZoneAdjustmentStub:
    """Minimal duck-type for ZoneAdjustment used in rate-limiting logic."""
    def __init__(self, zone_id: str, fan_speed_pct: float, supply_air_temp_setpoint_c: float):
        self.zone_id = zone_id
        self.fan_speed_pct = fan_speed_pct
        self.supply_air_temp_setpoint_c = supply_air_temp_setpoint_c


class _DCActionStub:
    """Minimal duck-type for DCAction used in rate-limiting logic."""
    def __init__(
        self,
        zone_adjustments: List[_ZoneAdjustmentStub],
        chiller_setpoint_c: float,
        chiller_active: bool,
        reasoning: Optional[str] = None,
    ):
        self.zone_adjustments = zone_adjustments
        self.chiller_setpoint_c = chiller_setpoint_c
        self.chiller_active = chiller_active
        self.reasoning = reasoning


# Re-export under the names used in type hints above
DCAction = _DCActionStub
ZoneAdjustment = _ZoneAdjustmentStub


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clip(value: float, max_abs: float) -> float:
    """Clip value to [-max_abs, +max_abs]."""
    return max(-max_abs, min(max_abs, value))


def _clip_bounds(value: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, value))