"""
HF Spaces training launcher for ClusterEnv GRPO.

Runs training in a background thread while exposing a minimal HTTP health
endpoint on port 7860 so HF doesn't mark the Space as unhealthy and kill it.

Training logs appear in the Space's build/runtime logs tab.
Checkpoints are pushed to HF Hub every 10 iterations via HF_HUB_REPO env var.
"""

import os
import sys
import threading
import http.server
import socketserver
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PORT = int(os.environ.get("PORT", 7860))

_status = {"state": "starting", "iteration": 0, "last_reward": 0.0}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (
            f"ClusterEnv GRPO Training\n"
            f"State    : {_status['state']}\n"
            f"Iteration: {_status['iteration']}\n"
            f"Reward   : {_status['last_reward']:+.4f}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress noisy access logs


def _health_server():
    with socketserver.TCPServer(("", PORT), _Handler) as httpd:
        httpd.serve_forever()


def _download_ppo_model() -> None:
    """
    Download the PPO cooling controller from HF Hub if not present locally.
    The file is too large to ship via git (binary), so we pull it at runtime.
    """
    dest = "/app/training/cooling_controller_best/best_model.zip"
    if os.path.exists(dest):
        print(f"PPO model already present at {dest}")
        return
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        hf_hub_download(
            repo_id   = "Mephisto2412/clusterenv-ppo-cooling",
            filename  = "best_model.zip",
            local_dir = "/app/training/cooling_controller_best",
        )
        print("PPO cooling model downloaded from HF Hub.")
    except Exception as e:
        print(f"PPO model download failed ({e}). CoolingHeuristic fallback will be used.")


def _run_training():
    _status["state"] = "downloading PPO model"
    _download_ppo_model()
    _status["state"] = "training"
    try:
        from training.train_grpo import main
        main()
        _status["state"] = "complete"
    except Exception as e:
        _status["state"] = f"error: {e}"
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(f"Starting health server on port {PORT} ...")
    health_thread = threading.Thread(target=_health_server, daemon=True)
    health_thread.start()

    print("Starting GRPO training ...")
    training_thread = threading.Thread(target=_run_training)
    training_thread.start()
    training_thread.join()

    print(f"Training finished with state: {_status['state']}")
    # Keep health server alive so HF doesn't recycle the Space before user reads logs
    while True:
        time.sleep(60)
