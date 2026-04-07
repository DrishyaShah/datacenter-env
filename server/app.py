# FastAPI server exposing the Data Centre OpenEnv environment (EnvClient-compatible).

from openenv.core.env_server.http_server import create_app

from .environment import DCEnvironment
from .models import DCAction, DCObservation

app = create_app(
    DCEnvironment,
    DCAction,
    DCObservation,
    env_name="datacenter_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the server locally: python -m datacenter_env.server.app or uv run server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # openenv validate checks for the substring "main()" in this module
    main(port=args.port)  # entry: main()
