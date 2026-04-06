# client.py — DC-OpenEnv Local Client
# Connects to your local datacenter_env FastAPI server

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DCAction, DCObservation  # <-- use your local models


class DCEnv(EnvClient[DCAction, DCObservation, State]):
    """
    Client for the DC-OpenEnv environment.
    Maintains a dedicated WebSocket session for efficient step/reset interactions.
    """

    def _step_payload(self, action: DCAction) -> Dict:
        """
        Convert DCAction to JSON payload for environment step call.
        """
        return {
            "zone_adjustments": [z.model_dump() for z in action.zone_adjustments],
            "chiller_setpoint_c": action.chiller_setpoint_c,
            "chiller_active": action.chiller_active,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DCObservation]:
        """
        Parse environment response JSON into StepResult[DCObservation].
        """
        obs_data = payload.get("observation", {})
        observation = DCObservation(**obs_data)  # assumes DCObservation fields match env output

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse environment state JSON into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
