from .models import (
    DCAction,
    DCObservation,
    ZoneAdjustment
    
)


class PIDBaseline:
    """
    Rule-based baseline controller for PUE benchmarking.
    Sets fan speed proportional to temperature deviation from setpoint.
    Sets chiller to fixed setpoint of 10C.
    This is what the agent's PUE is measured against.
    """
    def act(self, obs: DCObservation) -> DCAction:
        adjustments = []
        for zone in obs.zones:
            error = zone.cold_aisle_temp_c - zone.supply_air_temp_setpoint_c
            fan = 60.0 + max(0, error) * 8.0  # proportional
            fan = max(40.0, min(100.0, fan))
            adjustments.append(ZoneAdjustment(
                zone_id=zone.zone_id,
                fan_speed_pct=fan,
                supply_air_temp_setpoint_c=18.0,
            ))
        return DCAction(
            zone_adjustments=adjustments,
            chiller_setpoint_c=10.0,
            chiller_active=True,
        )