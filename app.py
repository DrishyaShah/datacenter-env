"""
ClusterEnv GRPO Training — Gradio Space launcher.

Starts training in a background thread immediately on Space startup.
Gradio keeps the Space healthy (port 7860 responding) while training runs.
Checkpoints are pushed to HF Hub every 10 iterations.
"""

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr

_status  = {"state": "starting", "iteration": 0, "reward": 0.0, "log": []}
_started = threading.Event()


def _log(msg: str) -> None:
    print(msg, flush=True)
    _status["log"].append(msg)
    if len(_status["log"]) > 200:
        _status["log"] = _status["log"][-200:]


def _download_ppo() -> None:
    dest = os.path.join(ROOT, "training", "cooling_controller_best", "best_model.zip")
    if os.path.exists(dest):
        _log("PPO model already present.")
        return
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        hf_hub_download(
            repo_id   = "Mephisto2412/clusterenv-ppo-cooling",
            filename  = "best_model.zip",
            local_dir = os.path.dirname(dest),
        )
        _log("PPO cooling model downloaded from HF Hub.")
    except Exception as e:
        _log(f"PPO download failed ({e}) — heuristic fallback active.")


def _run_training() -> None:
    _status["state"] = "downloading PPO model"
    _download_ppo()

    _status["state"] = "training"
    _log("Starting GRPO training...")
    try:
        from training.train_grpo import main
        main()
        _status["state"] = "complete"
        _log("Training complete.")
    except Exception as e:
        import traceback
        _status["state"] = f"error: {e}"
        _log(f"Training error: {e}")
        traceback.print_exc()


# Start training immediately in background
_thread = threading.Thread(target=_run_training, daemon=False)
_thread.start()


# ── Gradio interface (keeps Space healthy) ────────────────────────────────────

def get_status() -> tuple[str, str]:
    state = _status["state"]
    it    = _status["iteration"]
    rew   = _status["reward"]
    log   = "\n".join(_status["log"][-50:])  # last 50 lines
    header = f"State: {state} | Iteration: {it}/60 | Last reward: {rew:+.4f}"
    return header, log


with gr.Blocks(title="ClusterEnv GRPO Training") as demo:
    gr.Markdown(
        "## ClusterEnv GRPO Scheduler Training\n\n"
        "Training runs in the background. "
        "Checkpoints pushed to [Mephisto2412/clusterenv-grpo-adapter]"
        "(https://huggingface.co/Mephisto2412/clusterenv-grpo-adapter) "
        "every 10 iterations.\n\n"
        "Refresh this page to see updated status."
    )
    status_box = gr.Textbox(label="Status", lines=1, interactive=False)
    log_box    = gr.Textbox(label="Training log (last 50 lines)", lines=25,
                            interactive=False, max_lines=25)

    refresh_btn = gr.Button("Refresh logs")
    refresh_btn.click(fn=get_status, outputs=[status_box, log_box])

    # Auto-refresh every 30 seconds
    demo.load(fn=get_status, outputs=[status_box, log_box], every=30)


demo.launch(server_name="0.0.0.0", server_port=7860)
