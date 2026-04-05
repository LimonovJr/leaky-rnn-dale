"""
Visualisation: PCA trajectories, event-aligned time courses, dPCA plots.
All functions use target_onset / cue_onset keys (V3 naming).
"""

import numpy as np
import matplotlib.pyplot as plt

from src.analysis import (
    get_aligned_pca_segments, select_trials,
    compute_median_and_band, compute_mean_and_sem,
)


# ------------------------------------------------------------------ helpers

def _center_band(aligned, stat_mode, q_low, q_high):
    if stat_mode == "median":
        center, low, high = compute_median_and_band(aligned, q_low, q_high)
        band_label = f"{q_low}–{q_high} pct"
    else:
        center, sem = compute_mean_and_sem(aligned)
        low, high   = center - sem, center + sem
        band_label  = "SEM"
    return center, low, high, band_label


# ------------------------------------------------------------------ raw trajectory plots

def plot_pca_trajectories(trials, trial_proj, max_trials=20,
                          title="Hidden-state trajectories (PCA)", annotate_events=False):
    plt.figure(figsize=(7, 5))
    for i in range(min(max_trials, len(trial_proj))):
        traj, tr = trial_proj[i], trials[i]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        if annotate_events:
            plt.scatter(traj[0, 0], traj[0, 1], s=25)
            for key in ("cue_onset", "target_onset", "target_win_on", "model_resp_on"):
                idx = tr.get(key)
                if idx is not None and 0 <= idx < len(traj):
                    plt.scatter(traj[idx, 0], traj[idx, 1], s=25)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.show()


def plot_pca_trajectories_by_outcome(trials, trial_proj,
                                      outcomes=("correct", "abort", "miss"),
                                      max_per_group=10):
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 5), squeeze=False)
    for j, outcome in enumerate(outcomes):
        ax = axes[0, j]
        sub_t, sub_p = select_trials(trials, trial_proj=trial_proj, train_outcome=outcome)
        for traj in sub_p[:max_per_group]:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        ax.set_title(f"{outcome} | n={len(sub_p)}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout(); plt.show()


def plot_single_trial_pca(trials, trial_proj, i=0):
    traj, tr = trial_proj[i], trials[i]
    plt.figure(figsize=(7, 5))
    plt.plot(traj[:, 0], traj[:, 1], "-o", markersize=2)
    plt.scatter(traj[0, 0], traj[0, 1], label="start", s=60)
    for key, label in [("cue_onset", "cue"), ("target_onset", "target"),
                        ("target_win_on", "resp_win"), ("model_resp_on", "response")]:
        idx = tr.get(key)
        if idx is not None and 0 <= idx < len(traj):
            plt.scatter(traj[idx, 0], traj[idx, 1], label=label, s=60)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Trial {i} | {tr.get('train_outcome')} | ctoa_bin={tr.get('ctoa_bin')} | "
              f"distractors={tr.get('has_distractors')}")
    plt.legend(); plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ aligned plots

def plot_median_trajectory_pca(trials, trial_proj, align_key="target_onset",
                                window_before=40, window_after=40,
                                stat_mode="median", q_low=25, q_high=75, **filter_kw):
    aligned, rel_time, kept = get_aligned_pca_segments(
        trials, trial_proj, align_key=align_key,
        window_before=window_before, window_after=window_after, **filter_kw)
    if aligned is None:
        print("No trials matched."); return

    center, *_, _ = _center_band(aligned, stat_mode, q_low, q_high)
    zi = window_before

    plt.figure(figsize=(7, 5))
    plt.plot(center[:, 0], center[:, 1], "-o", markersize=4)
    plt.scatter(center[zi, 0], center[zi, 1], s=90, label=f"align: {align_key}")
    for i, t in enumerate(rel_time):
        if t in (-window_before, 0, window_after):
            plt.text(center[i, 0], center[i, 1], str(t), fontsize=9)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"{stat_mode} trajectory | {align_key} | n={len(kept)}")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_pc_timecourses(trials, trial_proj, align_key="target_onset",
                        window_before=40, window_after=40,
                        stat_mode="median", q_low=25, q_high=75, **filter_kw):
    aligned, rel_time, kept = get_aligned_pca_segments(
        trials, trial_proj, align_key=align_key,
        window_before=window_before, window_after=window_after, **filter_kw)
    if aligned is None:
        print("No trials matched."); return

    center, low, high, band_label = _center_band(aligned, stat_mode, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, dim, name in zip(axes, [0, 1], ["PC1", "PC2"]):
        ax.plot(rel_time, center[:, dim], label=f"{stat_mode} {name}")
        ax.fill_between(rel_time, low[:, dim], high[:, dim], alpha=0.25, label=band_label)
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{name} | n={len(kept)}")
        ax.set_xlabel(f"time from {align_key}"); ax.set_ylabel(name); ax.legend()
    plt.tight_layout(); plt.show()


def plot_two_group_median_trajectories(trials, trial_proj, align_key="target_onset",
                                        window_before=40, window_after=40,
                                        group1_kwargs=None, group2_kwargs=None,
                                        group1_label="group 1", group2_label="group 2"):
    g1k = group1_kwargs or {}
    g2k = group2_kwargs or {}
    shared = dict(align_key=align_key, window_before=window_before, window_after=window_after)

    a1, _, k1 = get_aligned_pca_segments(trials, trial_proj, **shared, **g1k)
    a2, _, k2 = get_aligned_pca_segments(trials, trial_proj, **shared, **g2k)
    if a1 is None: print(f"No trials for {group1_label}"); return
    if a2 is None: print(f"No trials for {group2_label}"); return

    med1, *_ = compute_median_and_band(a1)
    med2, *_ = compute_median_and_band(a2)
    zi = window_before

    plt.figure(figsize=(7, 5))
    plt.plot(med1[:, 0], med1[:, 1], "-o", markersize=4, label=f"{group1_label} | n={len(k1)}")
    plt.plot(med2[:, 0], med2[:, 1], "-o", markersize=4, label=f"{group2_label} | n={len(k2)}")
    plt.scatter(med1[zi, 0], med1[zi, 1], s=90)
    plt.scatter(med2[zi, 0], med2[zi, 1], s=90)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Median trajectories | {align_key}")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_two_group_pc_timecourses(trials, trial_proj, align_key="target_onset",
                                   window_before=40, window_after=40,
                                   group1_kwargs=None, group2_kwargs=None,
                                   group1_label="group 1", group2_label="group 2",
                                   stat_mode="median", q_low=25, q_high=75):
    g1k = group1_kwargs or {}
    g2k = group2_kwargs or {}
    shared = dict(align_key=align_key, window_before=window_before, window_after=window_after)

    a1, rt1, k1 = get_aligned_pca_segments(trials, trial_proj, **shared, **g1k)
    a2, rt2, k2 = get_aligned_pca_segments(trials, trial_proj, **shared, **g2k)
    if a1 is None: print(f"No trials for {group1_label}"); return
    if a2 is None: print(f"No trials for {group2_label}"); return

    c1, l1, h1, bl = _center_band(a1, stat_mode, q_low, q_high)
    c2, l2, h2, _  = _center_band(a2, stat_mode, q_low, q_high)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, dim, name in zip(axes, [0, 1], ["PC1", "PC2"]):
        ax.plot(rt1, c1[:, dim], label=group1_label)
        ax.fill_between(rt1, l1[:, dim], h1[:, dim], alpha=0.20)
        ax.plot(rt2, c2[:, dim], label=group2_label)
        ax.fill_between(rt2, l2[:, dim], h2[:, dim], alpha=0.20)
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{name} | {group1_label} n={len(k1)} | {group2_label} n={len(k2)}")
        ax.set_xlabel(f"time from {align_key}"); ax.set_ylabel(name); ax.legend()
    plt.tight_layout(); plt.show()


def plot_trialtype_panel(trials, trial_proj, group_specs, align_key="target_onset",
                          window_before=40, window_after=40, stat_mode="median",
                          q_low=25, q_high=75, figsize_per_row=(15, 4), suptitle=None):
    """
    Multi-group panel: one row per group — PCA plane, PC1(t), PC2(t).
    Each entry in group_specs is a dict with 'label' plus any filter kwargs.
    """
    n = len(group_specs)
    fig, axes = plt.subplots(n, 3,
                              figsize=(figsize_per_row[0], figsize_per_row[1] * n),
                              squeeze=False)

    for r, spec in enumerate(group_specs):
        spec = dict(spec)   # don't mutate caller's dict
        label = spec.pop("label", f"group {r+1}")

        aligned, rel_time, kept = get_aligned_pca_segments(
            trials, trial_proj, align_key=align_key,
            window_before=window_before, window_after=window_after, **spec)

        ax_traj, ax_pc1, ax_pc2 = axes[r]

        if aligned is None:
            for ax in (ax_traj, ax_pc1, ax_pc2):
                ax.set_title(f"{label} | n=0")
            continue

        center, low, high, band_label = _center_band(aligned, stat_mode, q_low, q_high)
        n_kept = aligned.shape[0]

        ax_traj.plot(center[:, 0], center[:, 1], "-o", markersize=3)
        ax_traj.scatter(center[window_before, 0], center[window_before, 1], s=80)
        ax_traj.set_title(f"{label} | n={n_kept}")
        ax_traj.set_xlabel("PC1"); ax_traj.set_ylabel("PC2")

        for ax, dim, name in [(ax_pc1, 0, "PC1"), (ax_pc2, 1, "PC2")]:
            ax.plot(rel_time, center[:, dim])
            ax.fill_between(rel_time, low[:, dim], high[:, dim], alpha=0.25, label=band_label)
            ax.axvline(0, linestyle="--")
            ax.set_title(f"{name} | n={n_kept}")
            ax.set_xlabel(f"time from {align_key}"); ax.set_ylabel(name); ax.legend()

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=14)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ dPCA plots

def plot_dpca_components(res, component_key="Z_cond", explained_key="explained_cond",
                          label_key="labels", n_plot=3, title_prefix="cond",
                          align_label="target_onset"):
    rel_time = res["rel_time"]
    Z        = res[component_key]
    labels   = res[label_key]

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharex=True)
    if n_plot == 1: axes = [axes]

    for k in range(n_plot):
        ax = axes[k]
        for i, lab in enumerate(labels):
            ax.plot(rel_time, Z[i, :, k], label=str(lab))
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{title_prefix}-dPC{k+1}")
        ax.set_xlabel(f"time from {align_label}"); ax.set_ylabel("component")

    axes[0].legend()
    plt.tight_layout(); plt.show()
    print(f"Explained variance ({title_prefix}):", res[explained_key][:n_plot])


def plot_dpca_plane(res, component_key="Z_cond", label_key="labels",
                    xlabel="cond-dPC1", ylabel="cond-dPC2",
                    title="Condition-demixed trajectories"):
    Z        = res[component_key]
    labels   = res[label_key]
    rel_time = res["rel_time"]
    zi       = int(np.where(rel_time == 0)[0][0])

    plt.figure(figsize=(7, 5))
    for i, lab in enumerate(labels):
        traj = Z[i]
        plt.plot(traj[:, 0], traj[:, 1], "-o", markersize=3, label=str(lab))
        plt.scatter(traj[zi, 0], traj[zi, 1], s=80)

    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()
