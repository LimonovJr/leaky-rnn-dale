"""
Training losses, regularisation, and the main supervised training loop.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# losses
# ------------------------------------------------------------------


def masked_temporal_cross_entropy(
    logits: torch.Tensor,   # [B, T, C]
    targets: torch.Tensor,  # [B, T]
    mask: torch.Tensor,     # [B, T]  scalar weight per timestep
) -> torch.Tensor:
    """
    Cross-entropy averaged over *weighted* timesteps.
    Padding positions should have mask=0.
    """
    B, T, C = logits.shape
    loss_per_step = F.cross_entropy(
        logits.reshape(B * T, C),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    return (loss_per_step * mask).sum() / mask.sum().clamp_min(1.0)


def l2_activity_and_weight_penalty(
    hidden_seq: torch.Tensor,  # [B, T, H]
    model: nn.Module,
    l2_h: float = 1e-6,
    l2_w: float = 1e-6,
) -> torch.Tensor:
    """L2 penalty on hidden activity and weight magnitudes."""
    reg = l2_h * hidden_seq.pow(2).mean()
    for name, p in model.named_parameters():
        if any(k in name for k in ("W_in", "W_rec", "W_out")):
            reg = reg + l2_w * p.pow(2).mean()
    return reg


# ------------------------------------------------------------------
# evaluation helpers
# ------------------------------------------------------------------


@torch.no_grad()
def decode_actions(logits: torch.Tensor) -> torch.Tensor:
    """
    logits : [B, T, C]
    returns: [B, T] int64 action indices
    """
    return logits.argmax(dim=-1)


@torch.no_grad()
def compute_trial_outcomes(
    pred_actions: torch.Tensor,   # [B, T]
    target_actions: torch.Tensor, # [B, T]
    lengths: torch.Tensor,        # [B]
) -> Dict[str, float]:
    """
    Classify each trial as correct / abort / miss.

    correct : first predicted release falls inside the true release window
    abort   : first predicted release falls outside the true release window
    miss    : no predicted release at all
    """
    stats: Dict[str, int] = {"correct": 0, "miss": 0, "abort": 0}

    for b in range(pred_actions.shape[0]):
        T = int(lengths[b].item())
        pa = pred_actions[b, :T].cpu().numpy()
        ta = target_actions[b, :T].cpu().numpy()

        release_pred = np.where(pa == 1)[0]

        if len(release_pred) == 0:
            stats["miss"] += 1
        else:
            t_rel = int(release_pred[0])
            stats["correct" if ta[t_rel] == 1 else "abort"] += 1

    total = sum(stats.values())
    return {k: v / total for k, v in stats.items()} if total > 0 else stats


# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    max_updates: int = 5000
    print_every: int = 50

    response_weight: float = 1.0
    baseline_weight: float = 1.0
    grace_ms: int = 0

    l2_h: float = 1e-6
    l2_w: float = 1e-6
    grad_clip: float = 10.0
    device: str = "cpu"


# ------------------------------------------------------------------
# training loop
# ------------------------------------------------------------------


def train_supervised(
    model: nn.Module,
    env_fn: Callable,
    cfg: TrainConfig,
) -> Dict[str, List[float]]:
    """
    Supervised training with masked temporal cross-entropy + L2 regularisation.

    Parameters
    ----------
    model  : BioLeakyRNN (or any nn.Module with the same forward signature)
    env_fn : zero-argument callable that returns a fresh environment instance
    cfg    : TrainConfig

    Returns
    -------
    history : dict with keys loss, ce, reg, p_correct, p_abort, p_miss
    """
    from src.dataset import make_train_batch  # local import to avoid circularity

    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    history: Dict[str, List[float]] = {
        "loss": [], "ce": [], "reg": [],
        "p_correct": [], "p_abort": [], "p_miss": [],
    }

    for upd in range(1, cfg.max_updates + 1):
        model.train()

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

        loss_ce = masked_temporal_cross_entropy(logits, y, mask)
        loss_reg = l2_activity_and_weight_penalty(h_seq, model, l2_h=cfg.l2_h, l2_w=cfg.l2_w)
        loss = loss_ce + loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        pred_actions = decode_actions(logits)
        stats = compute_trial_outcomes(pred_actions, y, lengths)

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

    return history
