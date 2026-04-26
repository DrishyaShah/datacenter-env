# FastAPI server exposing the Data Centre OpenEnv environment (EnvClient-compatible).

from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL Environment for Datacenter Cooling and Operations</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 48px auto;
           padding: 0 24px; color: #1a1a1a; background: #fafafa; }
    h1   { font-size: 1.6rem; margin-bottom: 4px; }
    .sub { color: #555; margin-bottom: 32px; }
    h2   { font-size: 1.1rem; margin-top: 28px; border-bottom: 1px solid #ddd;
           padding-bottom: 6px; }
    .ep  { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
    .tag { background: #e8f5e9; border: 1px solid #81c784; border-radius: 6px;
           padding: 6px 14px; font-size: 0.9rem; }
    .tag.red  { background: #fce4ec; border-color: #ef9a9a; }
    .tag.blue { background: #e3f2fd; border-color: #90caf9; }
    code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px;
           font-size: 0.9rem; }
    pre  { background: #f5f5f5; padding: 16px; border-radius: 8px;
           overflow-x: auto; font-size: 0.85rem; }
    a    { color: #1565c0; }
    .links a { display: inline-block; margin-right: 20px; }
  </style>
</head>
<body>
  <h1>RL Environment for Datacenter Cooling and Operations</h1>
  <p class="sub">OpenEnv-compliant environment &mdash; Theme #3.1 World Modeling &mdash; Hackathon Finale 2026</p>

  <h2>What this environment models</h2>
  <p>A shared AI compute cluster where two research teams compete for 900&nbsp;kW of power budget.
  One team submits honest job requests. The other systematically inflates priority, compresses
  deadlines, and hides carbon flexibility. An LLM scheduler (Qwen2.5-3B, GRPO-trained) learns
  to detect and discount misrepresentation while a pre-trained PPO controller manages the
  underlying thermal physics every step.</p>

  <div class="ep">
    <span class="tag">8 negotiation windows / episode</span>
    <span class="tag">18 physical steps / window</span>
    <span class="tag">900 kW hard power budget</span>
    <span class="tag blue">GRPO-trained LLM scheduler</span>
    <span class="tag red">PPO cooling controller</span>
  </div>

  <h2>OpenEnv HTTP API</h2>
  <pre>POST /reset          &larr; start a new episode, returns WindowState
POST /step           &larr; submit admission decisions, returns (WindowState, reward, done)
GET  /state          &larr; current environment state
GET  /health         &larr; liveness check</pre>

  <h2>Quick start</h2>
  <pre>from openenv import EnvClient

client = EnvClient("https://mephisto2412-datacenter-env.hf.space")
obs = client.reset()
print(obs)</pre>

  <h2>Links</h2>
  <p class="links">
    <a href="https://github.com/DrishyaShah/datacenter-env/tree/arhaan/finale-v1">GitHub Repo</a>
    <a href="https://huggingface.co/Mephisto2412/clusterenv-ppo-cooling">PPO Cooling Controller</a>
    <a href="https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/arhaan/finale-v1/training/train_grpo_colab.ipynb">Training Notebook</a>
  </p>
</body>
</html>"""


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
