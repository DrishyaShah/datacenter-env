"""
Rule-based proportional cooling controller for ClusterEnv.

Replaces the LLM-driven fan/setpoint decisions from Round 1 with a deterministic
heuristic. This is sufficient for v1: the physical consequences (temperatures rising
or falling) are what matter for the scheduler's training signal, not how the cooling
is managed. PPO-trained CoolingController is the v2 upgrade.

Usage pattern in ClusterEnvironment:
    from server.agents.cooling_heuristic import CoolingHeuristic

    heuristic  = CoolingHeuristic()
    prev_action = heuristic.initial_action(facility.zones)

    for step_idx in range(PHYSICAL_STEPS_PER_WINDOW):
        action      = heuristic.step(facility)
        facility.step(action, prev_action)   # prev_action for rate-limit delta
        prev_action = action
"""

from __future__ import annotations
from server.simulation import (
    FacilityState,
    ZoneState,
    _DCActionStub,
    _ZoneAdjustmentStub,
)


# -- Temperature thresholds ----------------------------------------------------
# Tuned for the 4-zone cluster scenario starting at 21.5-23.5C.
# The proportional bands ensure the heuristic responds before hitting 27C limit.

TEMP_AGGRESSIVE   = 26.0    # C -- ramp fans hard, drop setpoint aggressively
TEMP_WARM         = 24.5    # C -- moderate increase
TEMP_WATCH        = 23.0    # C -- gentle push
TEMP_NOMINAL_HIGH = 22.0    # C -- upper edge of hold band
TEMP_NOMINAL_LOW  = 20.5    # C -- lower edge of hold band; below = overcooling
TEMP_COLD         = 19.5    # C -- back off more aggressively

# Fan speed limits (%)
FAN_MAX    = 100.0
FAN_HIGH   = 90.0
FAN_HOLD   = 80.0
FAN_MIN    = 30.0    # never fully off; minimum for air circulation

# Supply temperature limits (C) -- must stay within [16, 26]
SUPPLY_MIN = 16.0
SUPPLY_MAX = 24.0


class CoolingHeuristic:
    """
    Proportional rule-based cooling controller.

    Stateless with respect to action history -- the ClusterEnvironment tracks
    the previous action for passing to FacilityState.step() as last_action
    (needed by apply_action_with_rate_limiting).

    step() is a pure function of the current FacilityState. Call it once per
    physical step, then pass both (action, prev_action) to facility.step().
    """

    def step(self, facility: FacilityState, upcoming_load_kw: list[float] | None = None) -> _DCActionStub:
        """
        Compute cooling action for the current facility state.

        Returns a _DCActionStub covering all zones plus facility-level chiller config.
        The caller is responsible for passing the previous action as last_action
        to facility.step() to correctly enforce rate limits.
        """
        zone_adjustments = []
        for zone in facility.zones:
            fan, setpoint = self._zone_action(zone)
            zone_adjustments.append(
                _ZoneAdjustmentStub(
                    zone_id                    = zone.zone_id,
                    fan_speed_pct              = fan,
                    supply_air_temp_setpoint_c = setpoint,
                )
            )

        return _DCActionStub(
            zone_adjustments   = zone_adjustments,
            chiller_setpoint_c = self._chiller_setpoint(facility),
            chiller_active     = facility.chiller_active,  # respect existing state
        )

    # -- Zone-level logic ------------------------------------------------------

    def _zone_action(self, zone: ZoneState) -> tuple[float, float]:
        """
        Compute (fan_speed_pct, supply_air_temp_setpoint_c) for one zone.

        Temperature bands:
          > TEMP_AGGRESSIVE (26C): emergency -- max fans, coldest supply
          > TEMP_WARM       (24.5C): warm -- strong increase
          > TEMP_WATCH      (23C): watch -- gentle push
          hold band         (20.5-22C): maintain current settings
          < TEMP_NOMINAL_LOW(20.5C): overcooling -- back off
          < TEMP_COLD       (19.5C): too cold -- reduce fans meaningfully
        """
        temp    = zone.temp_c
        fan     = zone.fan_speed_pct
        setpoint = zone.supply_air_temp_setpoint_c

        if temp > TEMP_AGGRESSIVE:
            # Emergency: push fans hard, drop supply temperature
            fan      = min(FAN_MAX, fan + 18.0)
            setpoint = max(SUPPLY_MIN, setpoint - 2.0)

        elif temp > TEMP_WARM:
            # Warm: ramp cooling meaningfully
            fan      = min(FAN_HIGH, fan + 10.0)
            setpoint = max(SUPPLY_MIN + 1.0, setpoint - 1.0)

        elif temp > TEMP_WATCH:
            # Heading warm: gentle nudge
            fan      = min(FAN_HOLD, fan + 5.0)
            setpoint = max(SUPPLY_MIN + 2.0, setpoint - 0.5)

        elif temp < TEMP_COLD:
            # Too cold: reduce fans significantly to save energy
            fan      = max(FAN_MIN, fan - 10.0)
            setpoint = min(SUPPLY_MAX, setpoint + 1.2)

        elif temp < TEMP_NOMINAL_LOW:
            # Slightly overcooling: minor back-off
            fan      = max(FAN_MIN + 10.0, fan - 5.0)
            setpoint = min(SUPPLY_MAX - 1.0, setpoint + 0.6)

        # else: temp in nominal hold band [20.5-22C] -- maintain current settings

        return round(fan, 1), round(setpoint, 1)

    # -- Facility-level logic --------------------------------------------------

    def _chiller_setpoint(self, facility: FacilityState) -> float:
        """
        Chiller setpoint based on mean zone temperature.
        Lower setpoint = colder chilled water = more cooling power but worse COP.
        Keep setpoint as warm as possible while still meeting zone needs.
        """
        if not facility.zones:
            return 10.0

        avg_temp = sum(z.temp_c for z in facility.zones) / len(facility.zones)

        if avg_temp > 25.5:
            return 7.0    # aggressive -- sacrifice COP for cooling
        elif avg_temp > 24.0:
            return 8.5    # warm -- moderate chilling
        elif avg_temp > 22.5:
            return 10.0   # nominal -- good COP
        else:
            return 11.0   # comfortable -- maximise COP

    # -- Factory for initial action --------------------------------------------

    @staticmethod
    def initial_action(zones: list[ZoneState]) -> _DCActionStub:
        """
        Create a safe starting action from current zone state.
        Pass this as last_action to the first facility.step() call so
        rate-limiting has a sensible reference point.
        """
        return _DCActionStub(
            zone_adjustments=[
                _ZoneAdjustmentStub(
                    zone_id                    = z.zone_id,
                    fan_speed_pct              = z.fan_speed_pct,
                    supply_air_temp_setpoint_c = z.supply_air_temp_setpoint_c,
                )
                for z in zones
            ],
            chiller_setpoint_c = 10.0,
            chiller_active     = True,
        )
