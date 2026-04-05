"""
Visualisation utilities for hidden-state analysis.

All functions follow a consistent API:
  - accept trials + trial_proj (PCA projections)
  - accept filtering kwargs forwarded to get_aligned_segments / select_trials
  - return None (display via plt.show)
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import (
    compute_mean_and_sem,
    compute_median_and_band,
    get_aligned_segments,
    select_trials,
)


# ------------------------------------------------------------------
# raw trajectory plots
# ------------------------------------------------------------------


def plot_pca_trajectories(
    trials: List[Dict],
    trial_proj: List[np.ndarray],
    max_trials: int = 20,
    title: str = "Hidden-state trajectories in PCA space",
    annotate_events: bool = False,
):
    """Plot first two PCs for up to *max_trials* trials."""
    plt.figure(figsize=(7, 5))
    n_plot = min(max_trials, len(trial_proj))

    for i in range(n_plot):
        traj = trial_proj[i]
        tr = trials[i]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.8)

        if annotate_events:
            plt.scatter(traj[0, 0], traj[0, 1], s=25)
            for key in ("cue_on", "target_on", "target_win_on", "model_resp_on"):
                idx = tr.get(key)
                if idx is not None and 0 <= idx < len(traj):
                    plt.scatter(traj[idx, 0], traj[idx, 1], s=25)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_single_trial_pca(
    trials: List[Dict],
    trial_proj: List[np.ndarray],
    i: int = 0,
):
    """Plot one trial in PCA space with labelled event markers."""
    traj = trial_proj[i]
    tr = trials[i]

    plt.figure(figsize=(7, 5))
    plt.plot(traj[:, 0], traj[:, 1], "-o", markersize=2)
    plt.scatter(traj[0, 0], traj[0, 1], label="start", s=60)

    for key, label in [
        ("cue_on", "cue_on"),
        ("target_on", "target_on"),
        ("target_win_on", "target_win_on"),
        ("model_resp_on", "model_resp_on"),
    ]:
        idx = tr.get(key)
        if idx is not None and 0 <= idx < len(traj):
            plt.scatter(traj[idx, 0], traj[idx, 1], label=label, s=60)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        f"Trial {i} | train={tr.get('train_outcome')} | "
        f"ctoa_bin={tr.get('ctoa_bin')} | distractors={tr.get('has_distractors')}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pca_trajectories_by_outcome(
    trials: List[Dict],
    trial_proj: List[np.ndarray],
    outcomes: Tuple[str, ...] = ("correct", "abort", "miss"),
    max_per_group: int = 10,
):
    """Plot separate subplots for each train outcome."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 5), squeeze=False)

    for j, outcome in enumerate(outcomes):
        ax = axes[0, j]
        sub_t, sub_p = select_trials(trials, trial_proj=trial_proj, train_outcome=outcome)
        for traj in sub_p[: max_per_group]:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        ax.set_title(f"{outcome} | n={len(sub_p)}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# event-aligned median / mean plots
# ------------------------------------------------------------------


def _get_center_band(aligned, stat_mode, q_low, q_high):
    if stat_mode == "median":
        center, low, high = compute_median_and_band(aligned, q_low=q_low, q_high=q_high)
        band_label = f"{q_low}–{q_high} pct"
    else:
        center, sem = compute_mean_and_sem(aligned)
        low, high = center - sem, center + sem
        band_label = "SEM"
    return center, low, high, band_label


def plot_median_trajectory_pca(
    trials, trial_proj,
    align_key="target_on",
    window_before=40, window_after=40,
    stat_mode="median", q_low=25, q_high=75,
    **filter_kwargs,
):
    aligned, rel_time, kept = get_aligned_segments(
        trials, trial_proj,
        align_key=align_key, window_before=window_before, window_after=window_after,
        **filter_kwargs,
    )
    if aligned is None:
        print("No trials matched.")
        return

    center, *_ , _ = _get_center_band(aligned, stat_mode, q_low, q_high)

    plt.figure(figsize=(7, 5))
    plt.plot(center[:, 0], center[:, 1], "-o", markersize=4)
    zi = window_before
    plt.scatter(center[zi, 0], center[zi, 1], s=90, label=f"align: {align_key}")
    for i, t in enumerate(rel_time):
        if t in (-window_before, 0, window_after):
            plt.text(center[i, 0], center[i, 1], str(t), fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{stat_mode.capitalize()} trajectory | align={align_key} | n={len(kept)}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pc_timecourses(
    trials, trial_proj,
    align_key="target_on",
    window_before=40, window_after=40,
    stat_mode="median", q_low=25, q_high=75,
    **filter_kwargs,
):
    """PC1(t) and PC2(t) as median+band or mean±SEM."""
    aligned, rel_time, kept = get_aligned_segments(
        trials, trial_proj,
        align_key=align_key, window_before=window_before, window_after=window_after,
        **filter_kwargs,
    )
    if aligned is None:
        print("No trials matched.")
        return

    center, low, high, band_label = _get_center_band(aligned, stat_mode, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, dim, label in zip(axes, [0, 1], ["PC1", "PC2"]):
        ax.plot(rel_time, center[:, dim], label=f"{stat_mode} {label}")
        ax.fill_between(rel_time, low[:, dim], high[:, dim], alpha=0.25, label=band_label)
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{label} | n={len(kept)}")
        ax.set_xlabel(f"time from {align_key}")
        ax.set_ylabel(label)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_two_group_pc_timecourses(
    trials, trial_proj,
    align_key="target_on",
    window_before=40, window_after=40,
    group1_kwargs=None, group2_kwargs=None,
    group1_label="group 1", group2_label="group 2",
    stat_mode="median", q_low=25, q_high=75,
):
    """Compare two groups as PC1(t) and PC2(t) time courses."""
    g1k = group1_kwargs or {}
    g2k = group2_kwargs or {}

    shared = dict(align_key=align_key, window_before=window_before, window_after=window_after)
    a1, rt1, k1 = get_aligned_segments(trials, trial_proj, **shared, **g1k)
    a2, rt2, k2 = get_aligned_segments(trials, trial_proj, **shared, **g2k)

    if a1 is None:
        print(f"No trials for {group1_label}")
        return
    if a2 is None:
        print(f"No trials for {group2_label}")
        return

    c1, l1, h1, bl = _get_center_band(a1, stat_mode, q_low, q_high)
    c2, l2, h2, _ = _get_center_band(a2, stat_mode, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, dim, pc_label in zip(axes, [0, 1], ["PC1", "PC2"]):
        ax.plot(rt1, c1[:, dim], label=f"{group1_label}")
        ax.fill_between(rt1, l1[:, dim], h1[:, dim], alpha=0.20)
        ax.plot(rt2, c2[:, dim], label=f"{group2_label}")
        ax.fill_between(rt2, l2[:, dim], h2[:, dim], alpha=0.20)
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{pc_label} | {group1_label} n={len(k1)} | {group2_label} n={len(k2)}")
        ax.set_xlabel(f"time from {align_key}")
        ax.set_ylabel(pc_label)
        ax.legend()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# panel plot (multi-group, 3 subplots per row)
# ------------------------------------------------------------------


def plot_trial_type_panel(
    trials, trial_proj,
    group_specs: List[Dict],
    align_key: str = "target_on",
    window_before: int = 40,
    window_after: int = 40,
    stat_mode: str = "median",
    q_low: int = 25, q_high: int = 75,
    figsize_per_row: Tuple[float, float] = (15, 4),
    suptitle: Optional[str] = None,
):
    """
    Multi-group panel: one row per group, three columns
    (PCA plane, PC1(t), PC2(t)).

    Each entry in *group_specs* is a dict with an optional 'label' key
    plus any filter kwargs accepted by get_aligned_segments.
    """
    n = len(group_specs)
    fig, axes = plt.subplots(
        n, 3,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n),
        squeeze=False,
    )

    for r, spec in enumerate(group_specs):
        label = spec.pop("label", f"group {r + 1}")
        aligned, rel_time, kept = get_aligned_segments(
            trials, trial_proj,
            align_key=align_key, window_before=window_before, window_after=window_after,
            **spec,
        )
        spec["label"] = label  # restore for caller

        ax_traj, ax_pc1, ax_pc2 = axes[r]

        if aligned is None:
            for ax in (ax_traj, ax_pc1, ax_pc2):
                ax.set_title(f"{label} | n=0 — no trials")
            continue

        center, low, high, band_label = _get_center_band(aligned, stat_mode, q_low, q_high)
        n_kept = aligned.shape[0]

        # trajectory
        ax_traj.plot(center[:, 0], center[:, 1], "-o", markersize=3)
        ax_traj.scatter(center[window_before, 0], center[window_before, 1], s=80)
        ax_traj.set_title(f"{label} | n={n_kept}")
        ax_traj.set_xlabel("PC1")
        ax_traj.set_ylabel("PC2")

        # PC1 / PC2 time courses
        for ax, dim, pc_name in [(ax_pc1, 0, "PC1"), (ax_pc2, 1, "PC2")]:
            ax.plot(rel_time, center[:, dim])
            ax.fill_between(rel_time, low[:, dim], high[:, dim], alpha=0.25, label=band_label)
            ax.axvline(0, linestyle="--")
            ax.set_title(f"{pc_name} | n={n_kept}")
            ax.set_xlabel(f"time from {align_key}")
            ax.set_ylabel(pc_name)
            ax.legend()

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=14)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# dPCA plots
# ------------------------------------------------------------------


def plot_dpca_components(
    res: Dict[str, Any],
    component_key: str = "Z_cond",
    explained_key: str = "explained_cond",
    label_key: str = "labels",
    n_plot: int = 3,
    title_prefix: str = "cond",
    align_label: str = "target_on",
):
    """
    Plot dPCA components as time courses (one subplot per component).

    Parameters
    ----------
    component_key : 'Z_time' or 'Z_cond'
    """
    rel_time = res["rel_time"]
    Z = res[component_key]           # [C, T, K]
    labels = res[label_key]

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for k in range(n_plot):
        ax = axes[k]
        for i, lab in enumerate(labels):
            ax.plot(rel_time, Z[i, :, k], label=str(lab))
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{title_prefix}-dPC{k + 1}")
        ax.set_xlabel(f"time from {align_label}")
        ax.set_ylabel("component")

    axes[0].legend()
    plt.tight_layout()
    plt.show()

    print(f"Explained variance ({title_prefix}):", res[explained_key][:n_plot])


def plot_dpca_plane(
    res: Dict[str, Any],
    component_key: str = "Z_cond",
    label_key: str = "labels",
    xlabel: str = "cond-dPC1",
    ylabel: str = "cond-dPC2",
    title: str = "Condition-demixed trajectories",
):
    Z = res[component_key]
    labels = res[label_key]
    rel_time = res["rel_time"]
    zero_idx = int(np.where(rel_time == 0)[0][0])

    plt.figure(figsize=(7, 5))
    for i, lab in enumerate(labels):
        traj = Z[i]
        plt.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, label=str(lab))
        plt.scatter(traj[zero_idx, 0], traj[zero_idx, 1], s=80)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
