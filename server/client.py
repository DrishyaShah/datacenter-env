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
            "zone_adjustments": action.zone_adjustments,
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

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# """Datacenter Env Environment Client."""

# from typing import Dict

# from openenv.core import EnvClient
# from openenv.core.client_types import StepResult
# from openenv.core.env_server.types import State

# from .server.models import DatacenterAction, DatacenterObservation


# class DatacenterEnv(
#     EnvClient[DatacenterAction, DatacenterObservation, State]
# ):
#     """
#     Client for the Datacenter Env Environment.

#     This client maintains a persistent WebSocket connection to the environment server,
#     enabling efficient multi-step interactions with lower latency.
#     Each client instance has its own dedicated environment session on the server.

#     Example:
#         >>> # Connect to a running server
#         >>> with DatacenterEnv(base_url="http://localhost:8000") as client:
#         ...     result = client.reset()
#         ...     print(result.observation.echoed_message)
#         ...
#         ...     result = client.step(DatacenterAction(message="Hello!"))
#         ...     print(result.observation.echoed_message)

#     Example with Docker:
#         >>> # Automatically start container and connect
#         >>> client = DatacenterEnv.from_docker_image("datacenter_env-env:latest")
#         >>> try:
#         ...     result = client.reset()
#         ...     result = client.step(DatacenterAction(message="Test"))
#         ... finally:
#         ...     client.close()
#     """

#     def _step_payload(self, action: DatacenterAction) -> Dict:
#         """
#         Convert DatacenterAction to JSON payload for step message.

#         Args:
#             action: DatacenterAction instance

#         Returns:
#             Dictionary representation suitable for JSON encoding
#         """
#         return {
#             "message": action.message,
#         }

#     def _parse_result(self, payload: Dict) -> StepResult[DatacenterObservation]:
#         """
#         Parse server response into StepResult[DatacenterObservation].

#         Args:
#             payload: JSON response data from server

#         Returns:
#             StepResult with DatacenterObservation
#         """
#         obs_data = payload.get("observation", {})
#         observation = DatacenterObservation(
#             echoed_message=obs_data.get("echoed_message", ""),
#             message_length=obs_data.get("message_length", 0),
#             done=payload.get("done", False),
#             reward=payload.get("reward"),
#             metadata=obs_data.get("metadata", {}),
#         )

#         return StepResult(
#             observation=observation,
#             reward=payload.get("reward"),
#             done=payload.get("done", False),
#         )

#     def _parse_state(self, payload: Dict) -> State:
#         """
#         Parse server response into State object.

#         Args:
#             payload: JSON response from state request

#         Returns:
#             State object with episode_id and step_count
#         """
#         return State(
#             episode_id=payload.get("episode_id"),
#             step_count=payload.get("step_count", 0),
#         )
