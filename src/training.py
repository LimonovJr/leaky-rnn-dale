"""
Losses and supervised training loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ losses

def masked_temporal_cross_entropy(logits, targets, mask):
    """logits [B,T,C], targets [B,T], mask [B,T] -> scalar"""
    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, C),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def l2_activity_and_weight_penalty(h_seq, model, l2_h=1e-6, l2_w=1e-6):
    """h_seq [B,T,H] -> scalar regularisation term"""
    reg = l2_h * h_seq.pow(2).mean()
    for name, p in model.named_parameters():
        if any(k in name for k in ("W_in", "W_rec", "W_out")):
            reg = reg + l2_w * p.pow(2).mean()
    return reg


# ------------------------------------------------------------------ eval helpers

@torch.no_grad()
def decode_actions(logits):
    """logits [B,T,C] -> actions [B,T]"""
    return logits.argmax(dim=-1)


@torch.no_grad()
def compute_trial_outcomes(pred_actions, target_actions, lengths):
    """
    Returns dict with fractional correct / abort / miss.
    correct: first release inside true release window
    abort:   first release outside window
    miss:    no release
    """
    stats = {"correct": 0, "miss": 0, "abort": 0}

    for b in range(pred_actions.shape[0]):
        T  = int(lengths[b])
        pa = pred_actions[b, :T].cpu().numpy()
        ta = target_actions[b, :T].cpu().numpy()

        first_release = np.where(pa == 1)[0]
        if len(first_release) == 0:
            stats["miss"] += 1
        else:
            t_rel = int(first_release[0])
            stats["correct" if ta[t_rel] == 1 else "abort"] += 1

    total = sum(stats.values())
    return {k: v / total for k, v in stats.items()} if total > 0 else stats


# ------------------------------------------------------------------ config

@dataclass
class TrainConfig:
    batch_size:      int   = 64
    lr:              float = 1e-3
    max_updates:     int   = 5000
    print_every:     int   = 50
    response_weight: float = 1.0
    baseline_weight: float = 1.0
    grace_ms:        int   = 0
    l2_h:            float = 1e-6
    l2_w:            float = 1e-6
    grad_clip:       float = 10.0
    device:          str   = "cpu"
    # early stopping: halt when p_miss==0 for this many consecutive updates
    # and restore weights from rollback_steps before the first zero-miss update.
    # set stop_on_no_miss=0 to disable.
    stop_on_no_miss:  int  = 3
    rollback_steps:   int  = 5
    warmup_updates:   int  = 500  # don't check early stopping for first N updates


# ------------------------------------------------------------------ train loop

def train_supervised(model, env_fn, cfg: TrainConfig):
    """
    Supervised training with masked temporal cross-entropy + L2 reg.

    Early stopping logic (when cfg.stop_on_no_miss > 0):
        - keeps a rolling buffer of the last rollback_steps checkpoints
        - when p_miss == 0 for stop_on_no_miss consecutive updates,
          restores weights from rollback_steps before the first zero-miss hit
        - this preserves miss trials in the final model

    Returns history dict (loss, ce, reg, p_correct, p_abort, p_miss).
    """
    import copy
    from collections import deque
    from src.dataset import make_train_batch

    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    history = {k: [] for k in ("loss", "ce", "reg", "p_correct", "p_abort", "p_miss")}

    # rolling buffer of state_dicts, length = rollback_steps
    # index 0 = oldest, -1 = most recent
    use_early_stop = cfg.stop_on_no_miss > 0
    checkpoint_buf = deque(maxlen=cfg.rollback_steps) if use_early_stop else None
    zero_miss_streak = 0   # how many consecutive updates had p_miss == 0
    first_zero_miss_upd = None

    for upd in range(1, cfg.max_updates + 1):
        model.train()

        # save checkpoint before this update (so buf[-1] is always current weights)
        if use_early_stop:
            checkpoint_buf.append(copy.deepcopy(model.state_dict()))

        x, y, mask, lengths = make_train_batch(
            env_fn=env_fn,
            batch_size=cfg.batch_size,
            dt=int(model.dt),
            response_weight=cfg.response_weight,
            baseline_weight=cfg.baseline_weight,
            grace_ms=cfg.grace_ms,
            device=cfg.device,
        )

        logits, _, h_seq = model(x, return_hidden=True)

        loss_ce  = masked_temporal_cross_entropy(logits, y, mask)
        loss_reg = l2_activity_and_weight_penalty(h_seq, model, cfg.l2_h, cfg.l2_w)
        loss     = loss_ce + loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        stats = compute_trial_outcomes(decode_actions(logits), y, lengths)

        history["loss"].append(float(loss))
        history["ce"].append(float(loss_ce))
        history["reg"].append(float(loss_reg))
        history["p_correct"].append(stats["correct"])
        history["p_abort"].append(stats["abort"])
        history["p_miss"].append(stats["miss"])

        if upd % cfg.print_every == 0 or upd == 1:
            print(
                f"Upd {upd:5d}/{cfg.max_updates} | "
                f"Loss {loss:.4f} | CE {loss_ce:.4f} | Reg {loss_reg:.4f} | "
                f"p_abort {stats['abort']:.2f}  p_miss {stats['miss']:.2f}  "
                f"p_corr {stats['correct']:.2f}"
            )

        # early stopping check (skip warmup period)
        if use_early_stop and upd > cfg.warmup_updates:
            if stats["miss"] == 0.0:
                if zero_miss_streak == 0:
                    first_zero_miss_upd = upd
                zero_miss_streak += 1
            else:
                zero_miss_streak = 0
                first_zero_miss_upd = None

            if zero_miss_streak >= cfg.stop_on_no_miss:
                # checkpoint_buf[0] is the oldest saved state — that's rollback_steps
                # before the current update (or as far back as we have)
                restore_state = checkpoint_buf[0]
                model.load_state_dict(restore_state)
                actual_rollback = min(len(checkpoint_buf), cfg.rollback_steps)
                print(
                    f"\nEarly stop at upd {upd}: p_miss==0 for {zero_miss_streak} "
                    f"consecutive updates (first at upd {first_zero_miss_upd}).\n"
                    f"Restored weights from {actual_rollback} steps before "
                    f"(upd ~{upd - actual_rollback})."
                )
                break

    return history
