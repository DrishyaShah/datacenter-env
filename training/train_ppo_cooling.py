"""
PPO pre-training script for the ClusterEnv cooling controller.

Trains a small MLP policy to control fan speeds and supply air temperatures
across 4 data centre zones. The trained policy is saved and loaded as a fixed
component during ClusterScheduler GRPO training.

Stack:
    stable-baselines3 PPO  ->  gymnasium CoolingGymEnv  ->  FacilityState physics

No LLM, no GRPO, no Person A code required.

Usage:
    cd datacenter-env
    python training/train_ppo_cooling.py

Output:
    training/cooling_controller_pretrained.zip    load with PPO.load(...)
    training/cooling_controller_best/             best checkpoint during eval
    training/ppo_cooling_reward_curve.png         commit this to the repo
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for Windows / headless
import matplotlib.pyplot as plt

# -- Make project root importable ---------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env

from training.gym_cooling_env import CoolingGymEnv


# -- Training configuration ----------------------------------------------------
TOTAL_TIMESTEPS  = 80_000    # 80k is sufficient -- flat zone fix removes undershoot trap
N_ENVS           = 4         # parallel environments (speeds up rollout collection)
EVAL_FREQ        = 10_000    # evaluate every N steps
N_EVAL_EPISODES  = 20        # episodes per evaluation run
CHECKPOINT_DIR   = "training/cooling_controller_best"
FINAL_MODEL_PATH = "training/cooling_controller_pretrained"
REWARD_PLOT_PATH = "training/ppo_cooling_reward_curve.png"

# PPO hyperparameters (tuned for this environment)
PPO_CONFIG = dict(
    learning_rate    = 3e-4,
    n_steps          = 512,     # rollout steps per env before each update
    batch_size       = 64,
    n_epochs         = 10,      # gradient steps per update
    gamma            = 0.99,    # discount factor (episodes are short, high gamma is fine)
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.01,    # small entropy bonus for exploration
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    policy_kwargs    = dict(net_arch=[128, 128]),  # 2-layer MLP (tiny, fast)
    verbose          = 1,
)


# -- Reward curve logger -------------------------------------------------------

class RewardLogger(BaseCallback):
    """Logs mean episode reward every EVAL_FREQ steps for plotting."""

    def __init__(self, eval_env, eval_freq, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self._eval_env       = eval_env
        self._eval_freq      = eval_freq
        self._n_eval         = n_eval_episodes
        self.steps_log:   list[int]   = []
        self.rewards_log: list[float] = []

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq == 0:
            rewards = []
            obs, _ = self._eval_env.reset()
            done = False
            ep_reward = 0.0
            for _ in range(self._n_eval * 18):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self._eval_env.step(action)
                ep_reward += r
                if term or trunc:
                    rewards.append(ep_reward)
                    ep_reward = 0.0
                    obs, _ = self._eval_env.reset()
            if rewards:
                mean_r = float(np.mean(rewards))
                self.steps_log.append(self.n_calls)
                self.rewards_log.append(mean_r)
                if self.verbose:
                    print(f"  [eval @ step {self.n_calls:>7,}] mean_reward = {mean_r:.4f}")
        return True


def save_reward_plot(steps, rewards, path: str) -> None:
    """Save the reward curve as a PNG committed to the repo."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, rewards, linewidth=2, color="#2196F3", label="PPO cooling controller")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="zero baseline")
    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Mean episode reward", fontsize=12)
    ax.set_title("PPO Cooling Controller -- Training Reward Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Reward curve saved -> {path}")


# -- Entry point ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("ClusterEnv PPO Cooling Controller -- Pre-training")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Eval frequency  : every {EVAL_FREQ:,} steps")
    print()

    # -- Step 1: Validate the environment against Gymnasium spec ---------------
    print("Validating environment...")
    debug_env = CoolingGymEnv(randomize_loads=False)  # deterministic for check
    check_env(debug_env, warn=True)
    debug_env.close()
    print("  Environment OK")
    print()

    # -- Step 2: Create vectorised training environments -----------------------
    train_env = make_vec_env(
        lambda: CoolingGymEnv(
            randomize_loads      = True,
            include_chiller_fault = False,   # no faults during initial training
        ),
        n_envs = N_ENVS,
    )

    eval_env = CoolingGymEnv(
        randomize_loads      = True,
        include_chiller_fault = False,
    )

    # -- Step 3: Build PPO model -----------------------------------------------
    model = PPO("MlpPolicy", train_env, **PPO_CONFIG)
    print(f"Model policy: {model.policy}")
    print(f"Parameters  : {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # -- Step 4: Callbacks -----------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    save_callback = EvalCallback(
        eval_env,
        best_model_save_path = CHECKPOINT_DIR,
        log_path             = CHECKPOINT_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        verbose              = 0,
    )

    reward_logger = RewardLogger(
        eval_env      = eval_env,
        eval_freq     = EVAL_FREQ,
        n_eval_episodes = N_EVAL_EPISODES,
        verbose       = 1,
    )

    # -- Step 5: Train ---------------------------------------------------------
    print("Starting training...")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [save_callback, reward_logger],
        progress_bar    = True,
    )
    print()
    print("Training complete.")

    # -- Step 6: Save final model ----------------------------------------------
    model.save(FINAL_MODEL_PATH)
    print(f"  Final model saved -> {FINAL_MODEL_PATH}.zip")

    best_path = os.path.join(CHECKPOINT_DIR, "best_model")
    print(f"  Best model saved -> {best_path}.zip")

    # -- Step 7: Save reward curve (must be committed to repo) -----------------
    if reward_logger.steps_log:
        save_reward_plot(
            reward_logger.steps_log,
            reward_logger.rewards_log,
            REWARD_PLOT_PATH,
        )
    print()

    # -- Step 8: Quick evaluation of trained vs. rule-based heuristic ---------
    from server.agents.cooling_heuristic import CoolingHeuristic
    from server.scenarios.cluster_scenario import build_cluster_facility

    print("Comparing trained PPO vs. rule-based heuristic (10 episodes each):")
    heuristic = CoolingHeuristic()

    ppo_rewards, heuristic_rewards = [], []

    for seed in range(10):
        # PPO episode
        obs, _ = eval_env.reset(seed=seed)
        ep_r = 0.0
        for _ in range(18):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(action)
            ep_r += r
            if term or trunc:
                break
        ppo_rewards.append(ep_r)

        # Heuristic episode (manual physics loop)
        facility = build_cluster_facility(seed=seed, window_idx=3)
        from server.agents.cooling_heuristic import CoolingHeuristic
        h         = CoolingHeuristic()
        prev      = CoolingHeuristic.initial_action(facility.zones)
        hr        = 0.0
        for step in range(18):
            action_h = h.step(facility)
            facility.step(action_h, prev)
            prev = action_h
            # Simple temp compliance reward for comparison
            temps = [z.temp_c for z in facility.zones]
            hr += sum(1.0 if 18 <= t <= 27 else -0.5 for t in temps) / len(temps)
        heuristic_rewards.append(hr / 18)

    print(f"  PPO mean reward:      {np.mean(ppo_rewards):.4f}  {np.std(ppo_rewards):.4f}")
    print(f"  Heuristic mean reward:{np.mean(heuristic_rewards):.4f}  {np.std(heuristic_rewards):.4f}")
    print()

    # Save comparison plot
    comp_path = "training/ppo_vs_heuristic.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Rule-based\nheuristic", "PPO trained"],
           [np.mean(heuristic_rewards), np.mean(ppo_rewards)],
           color=["#9E9E9E", "#2196F3"], width=0.4)
    ax.set_ylabel("Mean episode reward", fontsize=12)
    ax.set_title("Cooling Controller: Rule-based vs PPO", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved -> {comp_path}")
    print()
    print("Done. Commit the following files to the repo:")
    print(f"  {REWARD_PLOT_PATH}")
    print(f"  {comp_path}")
    print(f"  {FINAL_MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
