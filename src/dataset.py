"""
Trial rollout, target/mask building, batch padding.
"""

import numpy as np
import torch


def rollout_one_trial(env):
    """
    Read pre-generated trial from env.ob / env.gt without stepping.

    Returns:
        x        [T, obs_dim] float32
        gt       [T]          int64
        fa       [T]          float32  -- 1.0 inside distractor RT windows, else 0.0
        xy       [T, 2]       float32  -- target coords (STIM_POS[target_loc]),
                                         broadcast across time (constant per trial)
        xy_mask  [T]          float32  -- 1.0 from cue_onset to end of trial, else 0.0
                                         (auxiliary spatial loss is only meaningful
                                         after the cue has been presented)
    """
    env.reset()
    T = len(env.gt)
    if hasattr(env, "_fa_window"):
        fa = env._fa_window.astype(np.float32).copy()
    else:
        fa = np.zeros(T, dtype=np.float32)

    # prefer continuous position if available; fall back to discrete-index lookup
    if "target_pos" in env.trial:
        tx, ty = env.trial["target_pos"]
    else:
        tx, ty = env.STIM_POS[int(env.trial["target_loc"])]
    xy = np.zeros((T, 2), dtype=np.float32)
    xy[:, 0] = tx
    xy[:, 1] = ty

    # aux loss only after cue onset — no target info available during fixation
    cue_start = int(env.start_ind["cue"])
    xy_mask = np.zeros(T, dtype=np.float32)
    xy_mask[cue_start:] = 1.0

    return (
        env.ob.astype(np.float32).copy(),
        env.gt.astype(np.int64).copy(),
        fa,
        xy,
        xy_mask,
    )


def build_targets_and_mask(gt_seq, dt=20, response_weight=1.0, baseline_weight=1.0, grace_ms=0):
    """
    gt_seq [T] -> targets [T] int64, mask [T] float32.

    mask weights: baseline_weight everywhere, response_weight inside release window,
    0 for the first grace_ms of the release window.
    """
    T = len(gt_seq)
    targets = gt_seq.astype(np.int64).copy()
    mask = np.full(T, baseline_weight, dtype=np.float32)

    release_inds = np.where(gt_seq == 1)[0]
    if len(release_inds) > 0:
        resp_start = int(release_inds[0])
        resp_end   = int(release_inds[-1]) + 1
        mask[resp_start:resp_end] = response_weight

        if grace_ms > 0:
            grace_steps = int(round(grace_ms / dt))
            mask[resp_start : min(resp_start + grace_steps, resp_end)] = 0.0

    return targets, mask


def pad_batch(seqs, pad_value=0.0, dtype=np.float32):
    """seqs: list of [T_i, ...] -> padded [B, T_max, ...], lengths [B]"""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    T_max = int(lengths.max())
    out = np.full((len(seqs), T_max, *seqs[0].shape[1:]), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, lengths


def make_train_batch(env_fn, batch_size=64, dt=20, response_weight=1.0,
                     baseline_weight=1.0, grace_ms=0, device="cpu"):
    """
    Sample batch_size trials (one env per trial) and return tensors.

    Returns x [B, T, D], y [B, T], mask [B, T], fa [B, T],
            xy [B, T, 2], xy_mask [B, T], lengths [B].

    - fa marks distractor RT windows (1 where any release would be an FA).
    - xy is the target's (x, y) broadcast across time (constant per trial).
    - xy_mask is 1 from cue onset to end-of-trial (pad also 0).
    """
    xs, ys, masks, fas, xys, xy_masks = [], [], [], [], [], []

    for _ in range(batch_size):
        env = env_fn()
        x_seq, gt_seq, fa_seq, xy_seq, xy_mask_seq = rollout_one_trial(env)
        y_seq, m_seq = build_targets_and_mask(
            gt_seq, dt=dt,
            response_weight=response_weight,
            baseline_weight=baseline_weight,
            grace_ms=grace_ms,
        )
        xs.append(x_seq)
        ys.append(y_seq)
        masks.append(m_seq)
        fas.append(fa_seq)
        xys.append(xy_seq)
        xy_masks.append(xy_mask_seq)

    x_pad, lengths = pad_batch(xs, pad_value=0.0, dtype=np.float32)
    y_pad, _       = pad_batch(ys, pad_value=0,   dtype=np.int64)
    m_pad, _       = pad_batch(masks, pad_value=0.0, dtype=np.float32)
    fa_pad, _      = pad_batch(fas, pad_value=0.0, dtype=np.float32)
    xy_pad, _      = pad_batch(xys, pad_value=0.0, dtype=np.float32)
    xym_pad, _     = pad_batch(xy_masks, pad_value=0.0, dtype=np.float32)

    return (
        torch.tensor(x_pad,   dtype=torch.float32, device=device),
        torch.tensor(y_pad,   dtype=torch.long,    device=device),
        torch.tensor(m_pad,   dtype=torch.float32, device=device),
        torch.tensor(fa_pad,  dtype=torch.float32, device=device),
        torch.tensor(xy_pad,  dtype=torch.float32, device=device),
        torch.tensor(xym_pad, dtype=torch.float32, device=device),
        torch.tensor(lengths, dtype=torch.long,    device=device),
    )
