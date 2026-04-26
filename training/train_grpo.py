"""
GRPO training script for the ClusterEnv LLM scheduler.

Trains Llama-3.1-8B-Instruct (4-bit, LoRA r=16) to issue admission decisions
each negotiation window. Cooling is handled by the pre-trained PPO controller.

Stack:
    Unsloth FastLanguageModel  ->  ClusterEnvironment  ->  GRPO loss

Usage (HuggingFace Spaces A10G or local GPU):
    python training/train_grpo.py

Output:
    training/grpo_adapter/ckpt_{10,20,...}/   ← periodic LoRA checkpoints
    training/grpo_adapter/final/              ← final LoRA adapter
"""

from __future__ import annotations

import os
import sys
import warnings

# Suppress noisy-but-harmless transformer/unsloth warnings before any imports
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*use_return_dict.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Unsloth.*")

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.rollout import collect_rollouts, compute_grpo_advantages


# ── Configuration ─────────────────────────────────────────────────────────────

# Budget-aware config: Qwen2.5-3B fits on T4/A10G and costs ~$2-4/run vs ~$10/run for 8B.
# With $30 budget this gives 8-10 trials instead of 3. Switch back to 8B for final run.
MODEL_NAME        = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH    = 4096      # our prompts exceed 2048; do not lower this
LOAD_IN_4BIT      = True

LORA_R            = 16
LORA_ALPHA        = 32
LORA_TARGET_MODS  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

N_ITERATIONS      = 50        # 50 iters at ~3.5 hrs on L40S
G_EPISODES        = 2         # 16 samples/iter — half the calls, valid GRPO signal
LEARNING_RATE     = 1e-5
GRAD_CLIP         = 1.0
TEMPERATURE       = 0.7
MAX_NEW_TOKENS    = 512       # 6 decisions × ~40 tok = ~300 tok max needed; 512 is safe

CHECKPOINT_EVERY  = 10
ADAPTER_DIR       = os.path.join(ROOT, "training", "grpo_adapter")


# ── Model utilities ───────────────────────────────────────────────────────────


def load_model():
    """Load Llama-3.1-8B-Instruct with Unsloth + LoRA."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,       # auto: bfloat16 on A10G
        load_in_4bit   = LOAD_IN_4BIT,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGET_MODS,
        lora_dropout   = 0.0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )
    return model, tokenizer


def make_generate_fn(model, tokenizer, temperature: float = TEMPERATURE,
                     max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Return a generate_fn: str -> str closure.

    Switches to inference mode before generation; caller is responsible for
    switching back to training mode before the gradient step.
    """
    from unsloth import FastLanguageModel

    def generate_fn(prompt: str) -> str:
        FastLanguageModel.for_inference(model)
        inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens  = max_new_tokens,
                temperature     = temperature,
                do_sample       = True,
                pad_token_id    = tokenizer.eos_token_id,
                max_length      = inputs["input_ids"].shape[1] + max_new_tokens,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    return generate_fn


def compute_log_prob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
) -> torch.Tensor:
    """
    Compute sum of log-probabilities of completion tokens given prompt.

    Called during the gradient phase (model in training mode, gradients enabled).
    """
    full_text  = prompt + completion
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=MAX_SEQ_LENGTH,
    ).input_ids
    full_ids   = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=MAX_SEQ_LENGTH,
    ).input_ids.to(model.device)

    prompt_len = min(prompt_ids.shape[1], full_ids.shape[1] - 1)

    if full_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    outputs   = model(full_ids)
    log_probs = torch.log_softmax(outputs.logits[:, :-1], dim=-1)  # [1, seq-1, vocab]

    comp_ids = full_ids[0, prompt_len:]                            # [comp_len]
    if comp_ids.numel() == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    n_comp = min(comp_ids.shape[0], log_probs.shape[1] - prompt_len + 1)
    if n_comp <= 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    comp_lp = log_probs[0, prompt_len - 1: prompt_len - 1 + n_comp].gather(
        1, comp_ids[:n_comp].unsqueeze(1)
    ).squeeze(1)
    return comp_lp.sum()


# ── Plot helper ───────────────────────────────────────────────────────────────


def _save_training_plots(rewards: list[float], losses: list[float]) -> None:
    """Save reward and loss curves as PNG files committed to the repo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = list(range(1, len(rewards) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(iters, rewards, color="#2196F3", linewidth=2, marker="o", markersize=4)
        ax1.axhline(0.28, color="gray", linestyle="--", linewidth=1,
                    label="rule-based baseline (+0.28)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Mean episode reward")
        ax1.set_title("GRPO Reward Curve — ClusterEnv Scheduler")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(iters, losses, color="#F44336", linewidth=2, marker="o", markersize=4)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("GRPO loss")
        ax2.set_title("GRPO Loss Curve — ClusterEnv Scheduler")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(ROOT, "training", "grpo_training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Training curves saved -> {out_path}")
    except Exception as e:
        print(f"  Could not save plots: {e}")


# ── Hub push helper (non-fatal) ───────────────────────────────────────────────


def _try_push_to_hub(model, tokenizer, commit_tag: str) -> None:
    hf_repo  = os.environ.get("HF_HUB_REPO", "Mephisto2412/clusterenv-grpo-adapter")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_repo:
        return
    if not hf_token:
        print(f"  HF_TOKEN not set — skipping Hub push for {commit_tag}. "
              "Add HF_TOKEN in Space Settings → Variables and secrets.")
        return
    try:
        model.push_to_hub(hf_repo, token=hf_token, commit_message=commit_tag)
        tokenizer.push_to_hub(hf_repo, token=hf_token)
        print(f"  Pushed {commit_tag} -> {hf_repo}")
    except Exception as e:
        print(f"  Hub push failed ({e}) — local checkpoint retained, training continues.")


# ── Training loop ─────────────────────────────────────────────────────────────


def main() -> None:
    from unsloth import FastLanguageModel

    print("=" * 60)
    print("ClusterEnv GRPO Scheduler Training")
    print("=" * 60)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  LoRA r         : {LORA_R}  alpha={LORA_ALPHA}")
    print(f"  Iterations     : {N_ITERATIONS}")
    print(f"  Episodes/iter  : {G_EPISODES}  (-> {G_EPISODES * 8} samples/iter)")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Temperature    : {TEMPERATURE}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model, tokenizer = load_model()
    print(f"  Params (trainable): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    generate_fn = make_generate_fn(model, tokenizer)
    optimizer   = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr = LEARNING_RATE,
    )
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # ── Resume from latest checkpoint if one exists ───────────────────────────
    start_iteration = 0
    existing_ckpts = sorted([
        d for d in os.listdir(ADAPTER_DIR)
        if d.startswith("ckpt_") and os.path.isdir(os.path.join(ADAPTER_DIR, d))
    ], key=lambda x: int(x.split("_")[1]))
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        last_iter = int(last_ckpt.split("_")[1])
        ckpt_path = os.path.join(ADAPTER_DIR, last_ckpt)
        print(f"  Resuming from checkpoint {last_ckpt} ({ckpt_path})")
        from peft import PeftModel
        model.load_adapter(ckpt_path, adapter_name="default")
        start_iteration = last_iter
        print(f"  Starting from iteration {start_iteration + 1}/{N_ITERATIONS}")
    else:
        print("  No checkpoint found — starting from scratch.")
    print()

    # Load accumulated metrics from the most recent checkpoint (if any)
    reward_log: list[float] = []
    loss_log:   list[float] = []
    metrics_path = os.path.join(ADAPTER_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            saved = json.load(f)
        reward_log = saved.get("reward_log", [])
        loss_log   = saved.get("loss_log", [])
        print(f"  Loaded {len(reward_log)} prior metric entries from metrics.json")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iteration, N_ITERATIONS):
        base_seed = iteration * G_EPISODES

        # — Rollout phase (inference, no gradient) —
        rollouts   = collect_rollouts(
            generate_fn,
            n_episodes           = G_EPISODES,
            base_seed            = base_seed,
            enable_chiller_fault = True,
        )
        advantages = compute_grpo_advantages(rollouts)

        # — Training phase —
        FastLanguageModel.for_training(model)
        optimizer.zero_grad()
        total_loss = 0.0

        for sample, adv in zip(rollouts, advantages):
            lp   = compute_log_prob(
                model, tokenizer, sample["prompt"], sample["completion"]
            )
            loss = (-adv * lp) / len(rollouts)  # normalise by batch size
            loss.backward()
            total_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()

        # — Logging —
        rewards     = [r["reward"] for r in rollouts]
        mean_reward = float(np.mean(rewards))
        parse_fails = sum(1 for r in rollouts if r["reward"] <= -0.4)

        # Per-window group std (GRPO signal quality)
        win_groups: dict[int, list[float]] = defaultdict(list)
        for r in rollouts:
            win_groups[r["window_idx"]].append(r["reward"])
        mean_group_std = float(np.mean([np.std(v) for v in win_groups.values()]))

        print(
            f"[{iteration + 1:3d}/{N_ITERATIONS}]  "
            f"loss={total_loss:+.4f}  "
            f"reward={mean_reward:+.4f}  "
            f"group_std={mean_group_std:.3f}  "
            f"grad={grad_norm:.3f}  "
            f"parse_fail={parse_fails}/{len(rollouts)}"
        )

        reward_log.append(mean_reward)
        loss_log.append(total_loss)

        # Sample completion preview (first 3 iterations only)
        if iteration < 3:
            preview = rollouts[0]["completion"].replace("\n", " ")[:120]
            print(f"  sample: {preview!r}")

        # — Checkpoint —
        if (iteration + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(ADAPTER_DIR, f"ckpt_{iteration + 1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            with open(metrics_path, "w") as f:
                json.dump({"reward_log": reward_log, "loss_log": loss_log}, f)
            print(f"  Checkpoint saved -> {ckpt_path}")
            _try_push_to_hub(model, tokenizer, f"ckpt_{iteration + 1}")

    # ── Save final adapter ────────────────────────────────────────────────────
    final_path = os.path.join(ADAPTER_DIR, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(metrics_path, "w") as f:
        json.dump({"reward_log": reward_log, "loss_log": loss_log}, f)
    _try_push_to_hub(model, tokenizer, "final")
    # ── Save training curves as PNG (required by submission checklist) ────────
    _save_training_plots(reward_log, loss_log)

    print()
    print(f"Training complete. Final adapter -> {final_path}")
    print()
    print("Next steps:")
    print("  python scripts/demo_replay.py --generate --output demo_trained.json")
    print("  python scripts/demo_replay.py demo_baseline.json --compare demo_trained.json")


if __name__ == "__main__":
    main()
