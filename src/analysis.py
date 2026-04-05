"""
Post-hoc analysis utilities.

Functions
---------
collect_trials          — roll out model on env, collect hidden states + metadata
fit_pca_on_trials       — fit PCA on all hidden states, project per trial
select_trials           — filter trial list by metadata
get_aligned_segments    — cut event-aligned windows from PCA projections
compute_median_and_band — median + quantile band
compute_mean_and_sem    — mean ± SEM
dpca_marginals          — time- and condition-demixed PCA (dPCA-style)
collect_aligned_hidden_by_label — collect raw hidden segments grouped by condition
make_condition_mean_tensor      — average within conditions -> [C, T, H] tensor
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA


# ------------------------------------------------------------------
# trial collection
# ------------------------------------------------------------------


@torch.no_grad()
def collect_trials(
    model,
    env_fn: Callable,
    n_trials: int = 200,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Roll out *n_trials* using the model's greedy (argmax) policy and collect
    observations, hidden states, logits, decoded actions, and trial metadata.

    Returns
    -------
    list of dicts, each containing:
      x, h, y, a        — full-trial arrays
      cue_on, target_on, target_win_on, target_win_off, model_resp_on
      cue_loc, target_loc, ctoa_ms, ctoa_bin
      has_distractors, n_distractors, distractor_locs, distractor_cdoa_ms
      first_distractor_cdoa_ms
      train_outcome     — 'correct' / 'abort' / 'miss'
    """
    model.eval()
    _device = torch.device(device)
    trials = []

    for _ in range(n_trials):
        env = env_fn()
        env.reset()

        x_seq = env.ob.copy().astype(np.float32)   # [T, D]
        gt_seq = env.gt.copy().astype(np.int64)    # [T]

        x_t = torch.tensor(x_seq, dtype=torch.float32, device=_device).unsqueeze(0)
        logits, _, h_seq = model(x_t, return_hidden=True)

        logits_np = logits[0].cpu().numpy()          # [T, 2]
        h_np = h_seq[0].cpu().numpy()                # [T, H]
        actions = np.argmax(logits_np, axis=-1)      # [T]

        cue_on = int(env.start_ind["cue"])
        target_on = int(env.start_ind["target"])

        win_inds = np.where(gt_seq == 1)[0]
        target_win_on = int(win_inds[0]) if len(win_inds) > 0 else None
        target_win_off = int(win_inds[-1]) if len(win_inds) > 0 else None

        release_pred = np.where(actions == 1)[0]
        model_resp_on = int(release_pred[0]) if len(release_pred) > 0 else None

        if model_resp_on is None:
            train_outcome = "miss"
        elif gt_seq[model_resp_on] == 1:
            train_outcome = "correct"
        else:
            train_outcome = "abort"

        trials.append(
            {
                "x": x_seq,
                "h": h_np,
                "y": logits_np,
                "a": actions,
                "cue_on": cue_on,
                "target_on": target_on,
                "target_win_on": target_win_on,
                "target_win_off": target_win_off,
                "model_resp_on": model_resp_on,
                "cue_loc": env.trial.get("cue_loc"),
                "target_loc": env.trial.get("target_loc"),
                "ctoa_ms": env.trial.get("ctoa_ms"),
                "ctoa_bin": env.trial.get("ctoa_bin"),
                "has_distractors": env.trial.get("has_distractors", False),
                "n_distractors": env.trial.get("n_distractors", 0),
                "distractor_locs": env.trial.get("distractor_locs", []),
                "distractor_cdoa_ms": env.trial.get("distractor_cdoa_ms", []),
                "first_distractor_cdoa_ms": env.trial.get("first_distractor_cdoa_ms"),
                "train_outcome": train_outcome,
            }
        )

    return trials


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------


def fit_pca_on_trials(
    trials: List[Dict], n_components: int = 3
) -> Tuple[PCA, List[np.ndarray], np.ndarray]:
    """
    Fit PCA on concatenated hidden states from all trials.

    Returns
    -------
    pca           : fitted sklearn PCA object
    trial_proj    : list of [T_i, n_components] arrays
    explained_var : explained variance ratio [n_components]
    """
    all_h = np.concatenate([tr["h"] for tr in trials], axis=0)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(all_h)

    trial_proj = []
    start = 0
    for tr in trials:
        T = tr["h"].shape[0]
        trial_proj.append(Z[start : start + T])
        start += T

    return pca, trial_proj, pca.explained_variance_ratio_


# ------------------------------------------------------------------
# trial selection
# ------------------------------------------------------------------


def select_trials(
    trials: List[Dict],
    trial_proj: Optional[List[np.ndarray]] = None,
    train_outcome: Optional[str] = None,
    has_distractors: Optional[bool] = None,
    ctoa_bin_min: Optional[int] = None,
    ctoa_bin_max: Optional[int] = None,
    target_loc: Optional[int] = None,
):
    """
    Filter trials by metadata criteria.

    If *trial_proj* is given, returns (filtered_trials, filtered_proj);
    otherwise returns filtered_trials only.
    """
    keep = []
    for i, tr in enumerate(trials):
        ok = True
        if train_outcome is not None and tr.get("train_outcome") != train_outcome:
            ok = False
        if has_distractors is not None and tr.get("has_distractors") != has_distractors:
            ok = False
        if ctoa_bin_min is not None:
            if tr.get("ctoa_bin") is None or tr["ctoa_bin"] < ctoa_bin_min:
                ok = False
        if ctoa_bin_max is not None:
            if tr.get("ctoa_bin") is None or tr["ctoa_bin"] > ctoa_bin_max:
                ok = False
        if target_loc is not None and tr.get("target_loc") != target_loc:
            ok = False
        if ok:
            keep.append(i)

    filtered = [trials[i] for i in keep]
    if trial_proj is None:
        return filtered
    return filtered, [trial_proj[i] for i in keep]


# ------------------------------------------------------------------
# event-aligned windowing
# ------------------------------------------------------------------


def get_aligned_segments(
    trials: List[Dict],
    trial_proj: List[np.ndarray],
    align_key: str = "target_on",
    window_before: int = 40,
    window_after: int = 40,
    train_outcome: Optional[str] = None,
    has_distractors: Optional[bool] = None,
    ctoa_bin_min: Optional[int] = None,
    ctoa_bin_max: Optional[int] = None,
    target_loc: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict]]:
    """
    Cut event-aligned windows from PCA projections.

    Returns
    -------
    aligned      : [N, W, D] or None if no trials matched
    rel_time     : [W]       relative time in timesteps
    kept_trials  : list of trial dicts that were included
    """
    filtered, filtered_proj = select_trials(
        trials,
        trial_proj=trial_proj,
        train_outcome=train_outcome,
        has_distractors=has_distractors,
        ctoa_bin_min=ctoa_bin_min,
        ctoa_bin_max=ctoa_bin_max,
        target_loc=target_loc,
    )

    segments = []
    kept = []

    for tr, proj in zip(filtered, filtered_proj):
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        start = t0 - window_before
        end = t0 + window_after + 1
        if start < 0 or end > len(proj):
            continue
        segments.append(proj[start:end])
        kept.append(tr)

    if not segments:
        return None, None, []

    return (
        np.stack(segments, axis=0),
        np.arange(-window_before, window_after + 1),
        kept,
    )


# ------------------------------------------------------------------
# descriptive statistics
# ------------------------------------------------------------------


def compute_median_and_band(
    aligned: np.ndarray, q_low: int = 25, q_high: int = 75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    aligned : [N, W, D]
    Returns  median, low, high — each [W, D]
    """
    return (
        np.median(aligned, axis=0),
        np.percentile(aligned, q_low, axis=0),
        np.percentile(aligned, q_high, axis=0),
    )


def compute_mean_and_sem(
    aligned: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    aligned : [N, W, D]
    Returns  mean, sem — each [W, D]
    """
    mean = np.mean(aligned, axis=0)
    n = aligned.shape[0]
    sem = np.std(aligned, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


# ------------------------------------------------------------------
# dPCA-style decomposition
# ------------------------------------------------------------------


def dpca_marginals(
    X: np.ndarray, n_components: int = 3
) -> Dict[str, Any]:
    """
    Approximate dPCA via separate PCAs on the time-marginal and
    condition-marginal of X.

    Parameters
    ----------
    X : [C, T, H]  condition-averaged hidden states

    Returns
    -------
    dict with keys:
      Z_time, Z_cond        — projected trajectories [C, T, K]
      pca_time, pca_cond    — fitted PCA objects
      explained_time,
      explained_cond        — explained variance ratios [K]
      X_time, X_cond        — marginalised arrays [C, T, H]
    """
    C, T, H = X.shape

    grand = X.mean(axis=(0, 1), keepdims=True)          # [1, 1, H]
    X_time = np.repeat(X.mean(axis=0, keepdims=True) - grand, C, axis=0)  # [C, T, H]
    X_cond = X - grand - X_time                         # [C, T, H]

    pca_time = PCA(n_components=n_components)
    Z_time = pca_time.fit_transform(X_time.reshape(C * T, H)).reshape(C, T, n_components)

    pca_cond = PCA(n_components=n_components)
    Z_cond = pca_cond.fit_transform(X_cond.reshape(C * T, H)).reshape(C, T, n_components)

    return {
        "X_time": X_time,
        "X_cond": X_cond,
        "pca_time": pca_time,
        "pca_cond": pca_cond,
        "Z_time": Z_time,
        "Z_cond": Z_cond,
        "explained_time": pca_time.explained_variance_ratio_,
        "explained_cond": pca_cond.explained_variance_ratio_,
    }


def collect_aligned_hidden_by_label(
    trials: List[Dict],
    label_fn: Callable[[Dict], Any],
    align_key: str = "target_on",
    window_before: int = 40,
    window_after: int = 40,
) -> Tuple[Dict[Any, np.ndarray], np.ndarray]:
    """
    Group event-aligned hidden-state windows by a user-defined condition label.

    Parameters
    ----------
    label_fn : function(trial) -> hashable label (return None to skip)

    Returns
    -------
    by_label : dict[label] -> [N_label, W, H]
    rel_time : [W]
    """
    by_label: Dict[Any, list] = {}
    rel_time = np.arange(-window_before, window_after + 1)

    for tr in trials:
        label = label_fn(tr)
        if label is None:
            continue
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        h = tr["h"]
        start, end = t0 - window_before, t0 + window_after + 1
        if start < 0 or end > len(h):
            continue
        by_label.setdefault(label, []).append(h[start:end])

    return (
        {k: np.stack(v, axis=0) for k, v in by_label.items()},
        rel_time,
    )


def make_condition_mean_tensor(
    by_label: Dict[Any, np.ndarray],
    min_trials: int = 3,
) -> Tuple[Optional[np.ndarray], List[Any], List[int]]:
    """
    Average hidden states within each condition.

    Returns
    -------
    X      : [C, W, H] or None if no conditions survived
    labels : list of condition labels
    counts : list of trial counts per condition
    """
    labels, X_list, counts = [], [], []
    for label in sorted(by_label.keys(), key=str):
        arr = by_label[label]
        if arr.shape[0] < min_trials:
            continue
        labels.append(label)
        counts.append(arr.shape[0])
        X_list.append(arr.mean(axis=0))

    if not X_list:
        return None, [], []
    return np.stack(X_list, axis=0), labels, counts
