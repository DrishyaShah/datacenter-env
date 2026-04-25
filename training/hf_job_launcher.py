# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
HF Jobs launcher for ClusterEnv GRPO training.

Diagnostic run (5 iterations, ~25 min, ~$0.42):
    hf jobs uv run training/hf_job_launcher.py \
        --flavor a10g-small \
        --env N_ITERATIONS=5 \
        --env HF_HUB_REPO=DrishyaShah/clusterenv-grpo-adapter \
        --env HF_TOKEN=$HF_TOKEN \
        --detach

Full run (50 iterations, ~4 hrs, ~$4):
    hf jobs uv run training/hf_job_launcher.py \
        --flavor a10g-small \
        --env HF_HUB_REPO=DrishyaShah/clusterenv-grpo-adapter \
        --env HF_TOKEN=$HF_TOKEN \
        --detach
"""
import os
import subprocess
import sys


def run(cmd: list, **kwargs) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


# 1. Install git-lfs — PPO model zip (~500KB) is tracked via Git LFS
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "git-lfs"])
run(["git", "lfs", "install"])

# 2. Clone project from Drishya's branch
run(["git", "clone", "-b", "drishya/finale-v1",
     "https://github.com/DrishyaShah/datacenter-env.git"])
os.chdir("datacenter-env")

# 3. Guard: verify PPO zip is real content, not a 134-byte LFS pointer
zip_path = "training/cooling_controller_best/best_model.zip"
size = os.path.getsize(zip_path)
print(f"PPO zip size: {size:,} bytes", flush=True)
if size < 10_000:
    print("LFS pointer detected — pulling full object...", flush=True)
    run(["git", "lfs", "pull"])
    size = os.path.getsize(zip_path)
    assert size > 10_000, f"git lfs pull failed — zip still a pointer ({size} B)"

# 4. Install unsloth first (must come before transformers to avoid version conflict)
run(["uv", "pip", "install", "--quiet",
     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])

# 5. Install remaining deps (torch, transformers, sb3, matplotlib, etc.)
run(["uv", "pip", "install", "--quiet", "-r", "requirements.txt"])

# 6. Smoke test — fail fast before spending GPU time
run([sys.executable, "-c",
     "from unsloth import FastLanguageModel; "
     "from server.agents.ppo_cooling_controller import PPOCoolingController; "
     "from training.rollout import collect_rollouts; "
     "print('Smoke test: OK')"])

# 7. Run training — N_ITERATIONS env var is forwarded automatically by HF Jobs
run([sys.executable, "training/train_grpo.py"])
