# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DC-OpenEnv: Data Centre Operations Environment.

Fully OpenEnv-compliant environment.
Manages episodes, steps, rewards, and observations for a single-zone DC cooling task.
"""

import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment

from models import (
    DCObservation,
    DCAction,
    DCReward,
    StepResult,
    ResetResult,
    ZoneObservation,
    ZoneAdjustment,
)
from datacenter_env.server.simulation import FacilityState, build_easy_scenario
from grader_easy import EasyGraderState, compute_step_reward, compute_final_score

# ── Task configuration ─────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "easy-single-zone": {
        "description": "Single-zone temperature control under steady load",
        "max_steps": 12,
        "scenario_builder": build_easy_scenario,
    }
}


class DCEnvironment(Environment):
    """
    OpenEnv-compliant Data Centre environment.

    Implements reset(), step(), and state() for a simple single-zone cooling task.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy-single-zone", seed: Optional[int] = None):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Valid: {list(TASK_CONFIGS)}")
        self.task = task
        self.seed = seed
        self.config = TASK_CONFIGS[task]
        self.max_steps: int = self.config["max_steps"]

        self._facility: Optional[FacilityState] = None
        self._grader_state: Optional[EasyGraderState] = None
        self._step_count: int = 0
        self._done: bool = False
        self._episode_rewards = []

    # ── OpenEnv interface ──────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reset the environment and return initial observation."""
        seed = self.seed if self.seed is not None else random.randint(0, 99999)
        self._facility = self.config["scenario_builder"](seed=seed)
        self._grader_state = EasyGraderState()
        self._step_count = 0
        self._done = False
        self._episode_rewards = []

        obs = self._make_observation()
        return ResetResult(observation=obs, info={"task": self.task, "seed": seed})

    def step(self, action: DCAction) -> StepResult:
        """Apply agent action, advance simulation, compute reward, return StepResult."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._facility is None:
            raise RuntimeError("Call reset() before step().")

        # Apply fan speed adjustments
        self._apply_action(action)

        # Advance physics
        for zone in self._facility.zones:
            zone.step_thermal(self._facility.outside_temp_c)
        self._facility.advance_time()
        self._step_count += 1

        # Compute reward (easy task: single zone)
        zone = self._facility.zones[0]
        actual_temp = zone.temp_c
        step_reward, detail = compute_step_reward(
            zone_temp=actual_temp,
            current_pue=self._facility.pue,
            grader_state=self._grader_state,
        )
        self._episode_rewards.append(step_reward)

        done = self._step_count >= self.max_steps
        self._done = done

        reward_model = DCReward(
            total=step_reward,
            pue_reward=detail["pue_reward"],
            temperature_penalty=min(0, detail["temp_reward"]),
            humidity_penalty=0.0,
            breakdown=detail,
        )

        info = {}
        if done:
            final_score = compute_final_score(self._grader_state)
            info["final_score"] = final_score
            info["episode_rewards"] = self._episode_rewards

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=step_reward,
            reward_detail=reward_model,
            done=done,
            info=info,
        )

    def state(self) -> dict:
        """Return full internal state for debugging/inspection."""
        if self._facility is None:
            return {"status": "not_initialized"}
        return {
            "task": self.task,
            "step": self._step_count,
            "done": self._done,
            "facility": self._facility.to_observation_dict(),
            "grader": {
                "steps_in_range": self._grader_state.steps_in_range,
                "steps_total": self._grader_state.steps_total,
                "pue_readings": self._grader_state.pue_readings,
            } if self._grader_state else {},
            "episode_rewards": self._episode_rewards,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _apply_action(self, action: DCAction):
        """Apply fan speed adjustments to the FacilityState."""
        zone_map = {z.zone_id: z for z in self._facility.zones}
        for adj in action.zone_adjustments:
            if adj.zone_id in zone_map and adj.fan_speed_pct is not None:
                # Clamp to [0, 100]
                zone_map[adj.zone_id].fan_speed_pct = max(0.0, min(100.0, adj.fan_speed_pct))

    def _make_observation(self) -> DCObservation:
        """Convert FacilityState to DCObservation."""
        f = self._facility
        obs_dict = f.to_observation_dict()
        zones_obs = [ZoneObservation(**z) for z in obs_dict["zones"]]
        return DCObservation(
            step=obs_dict["step"],
            timestamp_hour=obs_dict["timestamp_hour"],
            outside_temp_c=obs_dict["outside_temp_c"],
            chiller_active=obs_dict["chiller_active"],
            chiller_cop=obs_dict["chiller_cop"],
            grid_carbon_intensity=obs_dict["grid_carbon_intensity"],
            current_pue=obs_dict["current_pue"],
            zones=zones_obs,
            maintenance_notes=obs_dict["maintenance_notes"],
            upcoming_events=obs_dict["upcoming_events"],
        )