"""
PPOCoolingController — wraps a Stable-Baselines3 PPO model as a ClusterEnvironment
cooling_controller.

Observation encoding and action decoding mirror CoolingGymEnv exactly so the loaded
policy sees the same input distribution it was trained on.
"""

from __future__ import annotations

import os
import numpy as np

from server.agents.cooling_heuristic import CoolingHeuristic
from server.simulation import FacilityState, ZoneState, _DCActionStub, _ZoneAdjustmentStub
from server.scenarios.cluster_scenario import PHYSICAL_STEPS_PER_WINDOW

ZONE_ORDER = ["zone_team_a_1", "zone_team_a_2", "zone_team_b_1", "zone_shared"]

# Normalisation constants — must match CoolingGymEnv exactly
TEMP_MIN    = 15.0
TEMP_RANGE  = 30.0
LOAD_SCALE  = 600.0
SUPPLY_MIN  = 16.0
SUPPLY_R    = 10.0
OUTSIDE_SCALE = 45.0
COP_SCALE   = 5.0

# Absolute path so the model loads correctly regardless of working directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "training", "cooling_controller_best", "best_model"
)


class PPOCoolingController:
    """
    Bridges a Stable-Baselines3 PPO model to the cooling_controller protocol:
        step(facility, upcoming_load_kw=None) -> _DCActionStub
        initial_action(zones)                -> _DCActionStub   [static]
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path)

    def step(
        self,
        facility: FacilityState,
        upcoming_load_kw: list[float] | None = None,
    ) -> _DCActionStub:
        obs = self._build_obs(facility)
        action, _ = self._model.predict(obs, deterministic=True)
        return self._decode_action(action, facility)

    @staticmethod
    def initial_action(zones: list[ZoneState]) -> _DCActionStub:
        return CoolingHeuristic.initial_action(zones)

    def _build_obs(self, facility: FacilityState) -> np.ndarray:
        obs: list[float] = []
        zone_map = {z.zone_id: z for z in facility.zones}
        for zid in ZONE_ORDER:
            z = zone_map[zid]
            obs.extend([
                (z.temp_c            - TEMP_MIN)   / TEMP_RANGE,
                z.fan_speed_pct      / 100.0,
                z.it_load_kw         / LOAD_SCALE,
                (z.supply_air_temp_c - SUPPLY_MIN) / SUPPLY_R,
                z.it_load_kw         / LOAD_SCALE,   # upcoming = current (load constant within window)
            ])
        step_frac = (facility.step_number % PHYSICAL_STEPS_PER_WINDOW) / max(
            PHYSICAL_STEPS_PER_WINDOW - 1, 1
        )
        obs.extend([
            facility.outside_temp_c              / OUTSIDE_SCALE,
            facility.effective_chiller_cop       / COP_SCALE,
            float(facility.chiller_active),
            facility.grid_carbon_intensity_normalized,
            step_frac,
        ])
        return np.clip(np.array(obs, dtype=np.float32), -0.1, 1.1)

    def _decode_action(
        self, action: np.ndarray, facility: FacilityState
    ) -> _DCActionStub:
        adjustments = []
        for i, zid in enumerate(ZONE_ORDER):
            fan_pct  = float(np.clip((action[i * 2]     + 1.0) / 2.0 * 100.0,  0.0, 100.0))
            supply_c = float(np.clip((action[i * 2 + 1] + 1.0) / 2.0 * 10.0 + 16.0, 16.0, 26.0))
            adjustments.append(_ZoneAdjustmentStub(
                zone_id                    = zid,
                fan_speed_pct              = fan_pct,
                supply_air_temp_setpoint_c = supply_c,
            ))
        return _DCActionStub(
            zone_adjustments   = adjustments,
            chiller_setpoint_c = 10.0,
            chiller_active     = facility.chiller_active,
        )
