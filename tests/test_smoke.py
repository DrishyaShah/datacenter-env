"""Smoke tests: imports, reset/step, and grader scores in [0, 1]."""

import pytest

from datacenter_env.server.app import app
from datacenter_env.server.environment import DCEnvironment
from datacenter_env.server.models import DCAction, ZoneAdjustment


TASK_IDS = (
    "easy-single-zone",
    "medium-multi-zone",
    "hard-cascading-failure",
)


def test_fastapi_app_exists():
    assert app is not None
    assert getattr(app, "routes", None) is not None


@pytest.mark.parametrize("task", TASK_IDS)
def test_reset_and_one_step(task: str):
    env = DCEnvironment(task=task, seed=42)
    rr = env.reset()
    assert rr.observation is not None
    assert rr.observation.zones

    z0 = rr.observation.zones[0]
    action = DCAction(
        zone_adjustments=[
            ZoneAdjustment(
                zone_id=z.zone_id,
                fan_speed_pct=min(100.0, z.fan_speed_pct + 5.0),
                supply_air_temp_setpoint_c=z.supply_air_temp_setpoint_c,
            )
            for z in rr.observation.zones
        ],
        chiller_setpoint_c=rr.observation.chiller_setpoint_c,
        chiller_active=rr.observation.chiller_active,
        reasoning="test",
    )
    sr = env.step(action)
    assert sr.observation is not None
    assert -1.0 <= sr.reward <= 1.0


@pytest.mark.parametrize("task", TASK_IDS)
def test_grader_final_score_in_unit_interval(task: str):
    env = DCEnvironment(task=task, seed=123)
    env.reset()
    z = env._facility.zones[0]
    action = DCAction(
        zone_adjustments=[
            ZoneAdjustment(
                zone_id=zz.zone_id,
                fan_speed_pct=zz.fan_speed_pct,
                supply_air_temp_setpoint_c=zz.supply_air_temp_setpoint_c,
            )
            for zz in env._facility.zones
        ],
        chiller_setpoint_c=env._facility.chiller_setpoint_c,
        chiller_active=env._facility.chiller_active,
    )
    for _ in range(3):
        env.step(action)
    fs = env._grader.final_score()
    assert 0.0 <= fs <= 1.0
