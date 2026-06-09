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

    Kept for backward compat with any analysis code that imports it. The
    training loop now uses masked_com_loss (no learnable mixer) instead.
    """
    sq = (pred_xy - true_xy).pow(2).sum(dim=-1)    # [B, T]
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


def masked_com_loss(h, coords, target_xy, mask, period=2.0):
    """
    Center-of-mass loss on a toroidal sheet.

        h          [B, T, H]   hidden activity (pre-ReLU; we clip below)
        coords     [H, 2]      fixed sheet coordinates of neurons
        target_xy  [B, T, 2]   target position (constant per trial, broadcast)
        mask       [B, T]      1 where COM-loss applies (post-cue window)
        period     float       toroidal period (2.0 for normalized [-1, +1])

    Computes the activity-weighted center of mass on the sheet:
        COM_d = sum_i (relu(h_i) * coords[i, d]) / sum_i relu(h_i)
    then forces COM to be close to the target position. Distance uses
    minimum-image wrap so neurons near a torus edge can drive COM toward
    a target on the opposite edge via the shorter periodic route.

    Unlike masked_spatial_mse + decode_xy, there is no learnable mixer.
    The network must literally place its activity bump at the target
    position to minimize this loss — there is no shortcut through
    arbitrary linear readout weights.
    """
    relu_h = F.relu(h)                                       # [B, T, H]
    mass = relu_h.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, T, 1]
    com_x = (relu_h @ coords[:, 0:1]) / mass                 # [B, T, 1]
    com_y = (relu_h @ coords[:, 1:2]) / mass                 # [B, T, 1]
    com = torch.cat([com_x, com_y], dim=-1)                  # [B, T, 2]
    diff = com - target_xy
    diff = diff - period * torch.round(diff / period)        # toroidal wrap
    sq = (diff ** 2).sum(dim=-1)                             # [B, T]
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


def sparsity_loss(h):
    """Mean post-ReLU activity. Pairs with masked_com_loss to encourage a
    tight, well-localized bump rather than a sheet-wide pedestal that
    happens to have its COM at the right place."""
    return F.relu(h).mean()


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
    # Spatial regularizers (replace the old aux_xy_loss):
    com_weight:        float = 0.5   # COM-loss: forces activity center of mass at target position
    sparsity_weight:   float = 0.01  # mean(ReLU(h)) penalty; keeps the bump tight
    aux_weight:        float = 0.0   # MSE of model.decode_xy(h) vs target_pos (= lambda_spatial);
                                     # 0 = off. Guarded by hasattr(model, "decode_xy").
    xy_mask_from:      str   = "cue" # "cue" or "target": when the spatial/aux loss becomes active
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

    history = {k: [] for k in ("loss", "ce", "reg", "fa_pen", "com", "sparsity", "aux",
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
            xy_mask_from=cfg.xy_mask_from,
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

        # COM-loss: activity center of mass must match target position on
        # the toroidal sheet. No learnable mixer — the network has to put
        # the actual bump where the target is. Requires model.coords [H, 2].
        if cfg.com_weight > 0.0 and hasattr(model, "coords"):
            loss_com = masked_com_loss(
                h_seq, model.coords, xy_true, xy_mask, period=2.0,
            )
        else:
            loss_com = torch.zeros((), device=logits.device)

        # Sparsity loss: keep the bump tight (otherwise a sheet-wide pedestal
        # with COM at target position would also satisfy COM-loss).
        if cfg.sparsity_weight > 0.0:
            loss_sparse = sparsity_loss(h_seq)
        else:
            loss_sparse = torch.zeros((), device=logits.device)

        # Auxiliary spatial readout MSE (= lambda_spatial): the separate decode_xy
        # head is supervised toward target_pos (xy_mask gates it to cue/target onset).
        # Linear readout, no Dale — a metric (x,y) decode parallel to go/no-go.
        if cfg.aux_weight > 0.0 and hasattr(model, "decode_xy"):
            loss_aux = masked_spatial_mse(model.decode_xy(h_seq), xy_true, xy_mask)
        else:
            loss_aux = torch.zeros((), device=logits.device)

        loss = (
            loss_ce
            + cfg.fa_penalty_weight * loss_fa
            + cfg.com_weight * loss_com
            + cfg.sparsity_weight * loss_sparse
            + cfg.aux_weight * loss_aux
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
        history["com"].append(loss_com.detach().item())
        history["sparsity"].append(loss_sparse.detach().item())
        history["aux"].append(loss_aux.detach().item())
        history["p_correct"].append(stats["correct"])
        history["p_abort"].append(stats["abort"])
        history["p_miss"].append(stats["miss"])
        history["fa_rate"].append(stats["fa_rate"])

        if upd % cfg.print_every == 0 or upd == 1:
            fa_str = f"{stats['fa_rate']*100:5.1f}%" if stats["n_distractor_trials"] > 0 else "  n/a"
            print(
                f"Upd {upd:5d}/{cfg.max_updates} | "
                f"Loss {loss.detach().item():.4f} | CE {loss_ce.detach().item():.4f} | "
                f"FA {loss_fa.detach().item():.4f} | "
                f"COM {loss_com.detach().item():.4f} | Sp {loss_sparse.detach().item():.4f} | "
                f"aux {loss_aux.detach().item():.4f} | "
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
