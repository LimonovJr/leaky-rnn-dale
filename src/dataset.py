"""
Trial rollout, target building, batch assembly.
"""

from typing import Callable, List, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------
# single-trial rollout
# ------------------------------------------------------------------


def rollout_one_trial(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll out one pre-generated trial using the hold action (action=0) at every
    step, collecting the full observation and ground-truth sequences.

    Returns
    -------
    x  : float32 array [T, obs_dim]
    gt : int64   array [T]
    """
    env.reset()
    # NeuroGym pre-generates the full trial on reset(), so we can read
    # env.ob and env.gt directly without stepping through the environment.
    x = env.ob.astype(np.float32)    # [T, obs_dim]
    gt = env.gt.astype(np.int64)     # [T]
    return x, gt


# ------------------------------------------------------------------
# target / mask building
# ------------------------------------------------------------------


def build_targets_and_mask(
    gt_seq: np.ndarray,
    dt: int = 20,
    response_weight: float = 1.0,
    baseline_weight: float = 1.0,
    grace_ms: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build integer class labels and a scalar temporal loss mask.

    Parameters
    ----------
    gt_seq          : [T] ground-truth action sequence (0=hold, 1=release)
    dt              : simulation timestep in ms
    response_weight : loss weight inside the response window
    baseline_weight : loss weight outside the response window
    grace_ms        : first <grace_ms> ms of the response window are masked out

    Returns
    -------
    targets : int64   [T]
    mask    : float32 [T]
    """
    T = len(gt_seq)
    targets = gt_seq.astype(np.int64).copy()
    mask = np.full(T, baseline_weight, dtype=np.float32)

    release_inds = np.where(gt_seq == 1)[0]
    if len(release_inds) > 0:
        resp_start = int(release_inds[0])
        resp_end = int(release_inds[-1]) + 1

        mask[resp_start:resp_end] = response_weight

        # optional grace period
        if grace_ms > 0:
            grace_steps = int(round(grace_ms / dt))
            g1 = min(resp_start + grace_steps, resp_end)
            mask[resp_start:g1] = 0.0

    return targets, mask


# ------------------------------------------------------------------
# padding
# ------------------------------------------------------------------


def pad_batch(
    seqs: List[np.ndarray],
    pad_value: float = 0.0,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad a list of variable-length arrays to the same length.

    Parameters
    ----------
    seqs : list of arrays with shape [T_i, ...]

    Returns
    -------
    padded  : [B, T_max, ...]
    lengths : int64 [B]
    """
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    T_max = int(lengths.max())
    tail_shape = seqs[0].shape[1:]
    out = np.full((len(seqs), T_max, *tail_shape), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lengths


# ------------------------------------------------------------------
# batch factory
# ------------------------------------------------------------------


def make_train_batch(
    env_fn: Callable,
    batch_size: int = 64,
    dt: int = 20,
    response_weight: float = 1.0,
    baseline_weight: float = 1.0,
    grace_ms: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a batch of trials and return tensors ready for training.

    A new environment instance is created for each trial so that
    trial-level randomisation is independent across the batch.

    Returns
    -------
    x       : float32 [B, T, obs_dim]
    y       : int64   [B, T]  class labels {0, 1}
    mask    : float32 [B, T]  per-timestep loss weights
    lengths : int64   [B]
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for _ in range(batch_size):
        env = env_fn()
        x_seq, gt_seq = rollout_one_trial(env)
        y_seq, m_seq = build_targets_and_mask(
            gt_seq,
            dt=dt,
            response_weight=response_weight,
            baseline_weight=baseline_weight,
            grace_ms=grace_ms,
        )
        xs.append(x_seq)
        ys.append(y_seq)
        masks.append(m_seq)

    x_pad, lengths = pad_batch(xs, pad_value=0.0, dtype=np.float32)
    y_pad, _ = pad_batch(ys, pad_value=0, dtype=np.int64)
    m_pad, _ = pad_batch(masks, pad_value=0.0, dtype=np.float32)

    return (
        torch.tensor(x_pad, dtype=torch.float32, device=device),
        torch.tensor(y_pad, dtype=torch.long, device=device),
        torch.tensor(m_pad, dtype=torch.float32, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )
