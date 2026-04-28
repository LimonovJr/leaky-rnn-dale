"""
Losses and supervised training loop.
"""

import copy
from collections import deque
from dataclasses import dataclass
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


def masked_spatial_mse(pred_xy, true_xy, mask):
    """
    Mean-squared-error spatial loss, masked over time.
        pred_xy  [B, T, 2]
        true_xy  [B, T, 2]
        mask     [B, T]       1 inside the window where aux loss applies
    Returns scalar mean per-timestep Euclidean-squared error.
    """
    sq = (pred_xy - true_xy).pow(2).sum(dim=-1)    # [B, T]
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


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
def compute_trial_outcomes(pred_actions, target_actions, lengths, fa_window=None):
    """
    Returns dict with fractional correct / abort / miss.
    correct: first release inside true release window
    abort:   first release outside window
    miss:    no release

    If fa_window is provided, also reports:
    fa_rate: fraction of distractor-bearing trials where any release
             landed inside a distractor FA window (NaN if no such trials).
    n_distractor_trials: count of trials with ≥1 distractor FA window.
    """
    stats = {"correct": 0, "miss": 0, "abort": 0}
    trials_with_distr = 0
    trials_with_fa    = 0

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

        if fa_window is not None:
            fa = fa_window[b, :T].cpu().numpy()
            if (fa > 0).any():
                trials_with_distr += 1
                if ((pa == 1) & (fa > 0)).any():
                    trials_with_fa += 1

    total = sum(stats.values())
    out = {k: v / total for k, v in stats.items()} if total > 0 else dict(stats)
    out["n_distractor_trials"] = trials_with_distr
    out["fa_rate"] = (trials_with_fa / trials_with_distr) if trials_with_distr > 0 else float("nan")
    return out


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
    fa_penalty_weight: float = 0.4   # weight on release-probability penalty inside distractor FA windows
    aux_xy_weight:   float = 0.1     # weight on auxiliary MSE loss: ||W_aux h - (x_target, y_target)||^2
    device:          str   = "cpu"
    # early stopping — all counts are in print steps (1 step = print_every updates)
    # set stop_on_no_miss=0 to disable entirely
    stop_on_no_miss:  int  = 3   # print steps with p_miss==0 in a row before stopping
    rollback_steps:   int  = 5   # how many print steps back to restore weights
    warmup_steps:     int  = 10  # ignore early stopping for first N print steps


# ------------------------------------------------------------------ train loop

def train_supervised(model, env_fn, cfg: TrainConfig):
    """
    Supervised training with masked temporal cross-entropy + L2 reg.

    Early stopping (when stop_on_no_miss > 0):
      - checked only on print steps (every print_every updates)
      - ignored for first warmup_steps print steps
      - when p_miss==0 for stop_on_no_miss consecutive print steps,
        restores weights from rollback_steps print steps before the first zero

    Returns history dict (loss, ce, reg, p_correct, p_abort, p_miss).
    """
    from src.dataset import make_train_batch

    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    history = {k: [] for k in ("loss", "ce", "reg", "fa_pen", "aux_xy",
                                "p_correct", "p_abort", "p_miss", "fa_rate")}

    use_early_stop = cfg.stop_on_no_miss > 0
    # buffer stores one checkpoint per print step
    checkpoint_buf  = deque(maxlen=cfg.rollback_steps) if use_early_stop else None
    zero_miss_streak = 0
    first_zero_miss_step = None
    print_step = 0   # counts how many print steps have happened

    for upd in range(1, cfg.max_updates + 1):
        model.train()

        x, y, mask, fa, xy_true, xy_mask, lengths = make_train_batch(
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

        # Explicit FA penalty: mean release-probability inside distractor FA windows
        fa_sum = fa.sum()
        if fa_sum > 0:
            p_release = F.softmax(logits, dim=-1)[..., 1]
            loss_fa = (p_release * fa).sum() / fa_sum
        else:
            loss_fa = torch.zeros((), device=logits.device)

        if cfg.aux_xy_weight > 0.0:
            xy_pred = model.decode_xy(h_seq)    # [B, T, 2]
            loss_aux = masked_spatial_mse(xy_pred, xy_true, xy_mask)
        else:
            loss_aux = torch.zeros((), device=logits.device)

        loss = (
            loss_ce
            + cfg.fa_penalty_weight * loss_fa
            + cfg.aux_xy_weight * loss_aux
            + loss_reg
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        stats = compute_trial_outcomes(decode_actions(logits), y, lengths, fa_window=fa)

        history["loss"].append(loss.detach().item())
        history["ce"].append(loss_ce.detach().item())
        history["reg"].append(loss_reg.detach().item())
        history["fa_pen"].append(loss_fa.detach().item())
        history["aux_xy"].append(loss_aux.detach().item())
        history["p_correct"].append(stats["correct"])
        history["p_abort"].append(stats["abort"])
        history["p_miss"].append(stats["miss"])
        history["fa_rate"].append(stats["fa_rate"])

        if upd % cfg.print_every == 0 or upd == 1:
            fa_str = f"{stats['fa_rate']*100:5.1f}%" if stats["n_distractor_trials"] > 0 else "  n/a"
            print(
                f"Upd {upd:5d}/{cfg.max_updates} | "
                f"Loss {loss.detach().item():.4f} | CE {loss_ce.detach().item():.4f} | "
                f"FA_pen {loss_fa.detach().item():.4f} | Aux {loss_aux.detach().item():.4f} | "
                f"hit {stats['correct']*100:5.1f}%  miss {stats['miss']*100:5.1f}%  abort {stats['abort']*100:5.1f}%  "
                f"FA {fa_str} (n={stats['n_distractor_trials']})"
            )

            if use_early_stop:
                print_step += 1
                checkpoint_buf.append(copy.deepcopy(model.state_dict()))

                # skip warmup period
                if print_step <= cfg.warmup_steps:
                    continue

                if stats["miss"] == 0.0:
                    if zero_miss_streak == 0:
                        first_zero_miss_step = print_step
                    zero_miss_streak += 1
                else:
                    zero_miss_streak = 0
                    first_zero_miss_step = None

                if zero_miss_streak >= cfg.stop_on_no_miss:
                    restore_state = checkpoint_buf[0]
                    model.load_state_dict(restore_state)
                    steps_back = len(checkpoint_buf) - 1
                    print(
                        f"\nEarly stop at upd {upd} (print step {print_step}): "
                        f"p_miss==0 for {zero_miss_streak} consecutive print steps.\n"
                        f"Restored weights from {steps_back} print steps back "
                        f"(upd ~{upd - steps_back * cfg.print_every})."
                    )
                    break

    return history
