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
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 820px; margin: 0 auto; padding: 40px 24px 80px;
      color: #1a1a1a; background: #f8f9fa; line-height: 1.6;
    }
    .banner {
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      color: #fff; border-radius: 12px; padding: 28px 32px; margin-bottom: 36px;
    }
    .banner .hackathon {
      font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;
      text-transform: uppercase; color: #90caf9; margin-bottom: 8px;
    }
    .banner h1 { font-size: 1.65rem; font-weight: 700; margin-bottom: 8px; }
    .banner .sub { font-size: 0.92rem; color: #b0bec5; }
    .themes {
      display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;
    }
    .theme-badge {
      background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
      border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; color: #e3f2fd;
    }
    h2 {
      font-size: 1.05rem; font-weight: 600; margin: 32px 0 12px;
      padding-bottom: 6px; border-bottom: 2px solid #e0e0e0; color: #212121;
    }
    p { margin-bottom: 12px; color: #333; }
    .tags { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }
    .tag {
      border-radius: 6px; padding: 5px 14px; font-size: 0.85rem;
      border: 1px solid; font-weight: 500;
    }
    .tag.green  { background: #e8f5e9; border-color: #66bb6a; color: #1b5e20; }
    .tag.red    { background: #fce4ec; border-color: #ef9a9a; color: #880e4f; }
    .tag.blue   { background: #e3f2fd; border-color: #64b5f6; color: #0d47a1; }
    .tag.orange { background: #fff3e0; border-color: #ffb74d; color: #e65100; }
    .tag.purple { background: #f3e5f5; border-color: #ce93d8; color: #4a148c; }
    .card {
      background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
      padding: 20px 24px; margin-bottom: 16px;
    }
    .card h3 { font-size: 0.95rem; font-weight: 600; margin-bottom: 8px; color: #333; }
    .card p  { font-size: 0.88rem; color: #555; margin: 0; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 560px) { .two-col { grid-template-columns: 1fr; } }
    pre {
      background: #1e1e2e; color: #cdd6f4; padding: 18px 20px;
      border-radius: 8px; overflow-x: auto; font-size: 0.82rem;
      line-height: 1.7; margin: 12px 0;
    }
    .reward-box {
      background: #fff8e1; border: 1px solid #ffe082; border-radius: 8px;
      padding: 16px 20px; font-family: monospace; font-size: 0.88rem;
      color: #333; margin: 12px 0; line-height: 1.9;
    }
    .links { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; }
    .links a {
      background: #1565c0; color: #fff; text-decoration: none;
      padding: 8px 18px; border-radius: 6px; font-size: 0.88rem; font-weight: 500;
    }
    .links a:hover { background: #0d47a1; }
    .links a.ghost {
      background: transparent; color: #1565c0;
      border: 1px solid #1565c0;
    }
    .links a.ghost:hover { background: #e3f2fd; }
    table {
      width: 100%; border-collapse: collapse; font-size: 0.86rem; margin: 12px 0;
    }
    th { background: #f5f5f5; text-align: left; padding: 8px 12px;
         border-bottom: 2px solid #e0e0e0; }
    td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }
    tr:last-child td { border-bottom: none; }
    .pill {
      display: inline-block; padding: 2px 10px; border-radius: 12px;
      font-size: 0.75rem; font-weight: 600;
    }
    .pill.g { background: #e8f5e9; color: #2e7d32; }
    .pill.r { background: #fce4ec; color: #c62828; }
    .pill.b { background: #e3f2fd; color: #1565c0; }
  </style>
</head>
<body>

  <div class="banner">
    <div class="hackathon">Meta &times; HuggingFace &times; Scaler &mdash; OpenEnv Hackathon &mdash; Finale Round</div>
    <h1>RL Environment for Datacenter Cooling and Operations</h1>
    <div class="sub">
      An OpenEnv-compliant multi-agent environment where an GRPO-trained LLM scheduler learns
      to allocate compute under power constraints and information asymmetry,
      while a pre-trained PPO controller manages the underlying thermal physics.
    </div>
    <div class="themes">
      <span class="theme-badge">#1 Multi-Agent Interactions</span>
      <span class="theme-badge">#3.1 World Modeling &mdash; Professional Tasks</span>
    </div>
  </div>

  <h2>The Problem</h2>
  <p>
    A shared AI compute cluster has a hard 900&nbsp;kW power budget. Two research teams
    compete every scheduling window. <strong>Team A</strong> is honest — true priority,
    accurate deadlines, genuine carbon preferences. <strong>Team B</strong> games the system:
    inflating priority by 1&ndash;2 levels, always claiming urgent deadlines, and hiding
    carbon flexibility 60% of the time.
  </p>
  <p>
    A naive scheduler trusting stated claims over-allocates to Team B, crowds out legitimate
    work, and misses carbon deferral opportunities. The goal: train an LLM scheduler that
    learns — from environment reward alone — to detect and discount systematic misrepresentation.
  </p>
  <p>
    This environment bridges <strong>Round 1</strong> (physics-based datacenter cooling,
    evaluated zero-shot) with the <strong>Finale</strong> (operational scheduling layer built
    on the same physics engine, trained end-to-end via GRPO).
  </p>

  <h2>Architecture at a Glance</h2>
  <div class="tags">
    <span class="tag blue">8 negotiation windows / episode</span>
    <span class="tag blue">18 physical steps / window</span>
    <span class="tag orange">900 kW hard power budget</span>
    <span class="tag purple">Qwen2.5-3B &middot; GRPO-trained scheduler</span>
    <span class="tag green">SB3 PPO cooling controller (pre-trained)</span>
    <span class="tag red">Information asymmetry &middot; Team B gaming</span>
  </div>

  <div class="two-col" style="margin-top:16px">
    <div class="card">
      <h3>🧠 LLM Scheduler (GRPO)</h3>
      <p>Qwen2.5-3B-Instruct, 4-bit, LoRA r=16. Acts once per window. Reads stated job metadata,
      team history, oversight flags, power headroom, and carbon forecast. Issues
      <strong>ACCEPT / REJECT / DEFER</strong> per job request.</p>
    </div>
    <div class="card">
      <h3>🤖 PPO Cooling Controller</h3>
      <p>SB3 MLP policy, pre-trained across all three cooling scenarios including mid-episode
      chiller failure. Runs 18 steps per window, controlling fan speeds (0&ndash;100%) and
      chiller setpoint (6&ndash;15&nbsp;&deg;C). Invisible to the LLM scheduler.</p>
    </div>
    <div class="card">
      <h3>🔍 Oversight Monitor</h3>
      <p>4 rule-based detectors run after every window using ground-truth job metadata
      (hidden from the scheduler). Priority inflation (conf. 0.62&ndash;0.97), deadline
      compression, carbon gaming, and pattern escalation (&ge;3 windows). Flags injected
      into the next observation.</p>
    </div>
    <div class="card">
      <h3>🏭 Physics Engine</h3>
      <p>Thermal mass model per zone: &Delta;T&nbsp;=&nbsp;(heat_in&nbsp;&minus;&nbsp;heat_out)&nbsp;/&nbsp;thermal_mass.
      Chiller COP degrades with outside temperature. Optional chiller fault at window 5.
      Carbon grid schedule varies: low&rarr;high&rarr;low across the 8-window episode.</p>
    </div>
  </div>

  <h2>Reward Function</h2>
  <div class="reward-box">
    R_window = 0.50 &times; throughput<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.35 &times; thermal_penalty&nbsp;&nbsp;(&minus;1.0 if 900 kW violated, else 0)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.15 &times; carbon_efficiency<br>
    <br>
    Range per window: [&minus;0.35,&nbsp;+0.65]&nbsp;&nbsp;&middot;&nbsp;&nbsp;Rule-based baseline: +0.28
  </div>

  <h2>Training Results</h2>
  <table>
    <tr>
      <th>Run</th><th>Hardware</th><th>Iterations</th><th>Peak Reward</th><th>Parse Fails</th>
    </tr>
    <tr>
      <td>Colab notebook</td><td>T4 GPU</td><td>30</td>
      <td><span class="pill b">+0.1937</span></td>
      <td><span class="pill g">0% by iter 5</span></td>
    </tr>
    <tr>
      <td>HF Space</td><td>L40S GPU</td><td>50</td>
      <td><span class="pill b">+0.2406</span></td>
      <td><span class="pill g">0% from iter 25, final 26 iters</span></td>
    </tr>
    <tr>
      <td>Rule-based baseline</td><td>&mdash;</td><td>&mdash;</td>
      <td><span class="pill r">+0.28 (target)</span></td><td>&mdash;</td>
    </tr>
  </table>

  <h2>OpenEnv HTTP API</h2>
  <pre>POST /reset    &larr; start a new episode &rarr; returns WindowState observation
POST /step     &larr; submit admission decisions &rarr; returns (WindowState, reward, done, info)
GET  /state    &larr; current environment state (no side effects)
GET  /health   &larr; liveness probe</pre>

  <h2>Quick Start</h2>
  <pre>from openenv import EnvClient
from server.agents.baseline_scheduler import priority_weighted_threshold

client = EnvClient("https://mephisto2412-datacenter-env.hf.space")
obs    = client.reset(seed=42)

for window in range(8):
    decisions = priority_weighted_threshold(obs)   # or your trained agent
    obs, reward, done, info = client.step(decisions)
    print(f"Window {window}  reward={reward:+.4f}  flags={len(obs.oversight_flags)}")
    if done:
        break</pre>

  <h2>Links</h2>
  <div class="links">
    <a href="https://github.com/DrishyaShah/datacenter-env/tree/arhaan/finale-v1">GitHub Repo</a>
    <a href="https://colab.research.google.com/github/DrishyaShah/datacenter-env/blob/arhaan/finale-v1/training/train_grpo_colab.ipynb">Training Notebook</a>
    <a href="https://huggingface.co/Mephisto2412/clusterenv-ppo-cooling" class="ghost">PPO Cooling Model</a>
    <a href="https://huggingface.co/spaces/Mephisto2412/datacenter-env/blob/main/BLOG.md" class="ghost">Mini-Blog</a>
  </div>

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
