"""
Visualisation: PCA trajectories, event-aligned time courses, dPCA plots.
All functions use target_onset / cue_onset keys (V3 naming).
"""

import numpy as np
import matplotlib.pyplot as plt

from src.analysis import (
    get_aligned_pca_segments, select_trials,
    compute_median_and_band, compute_mean_and_sem,
    prepare_jpca_input, fit_jpca, jpca_permutation_test,
    compute_tangling, tangling_by_ctoa_bin, polynomial_regression,
    decode_position_by_ctoa_bin,
)


def _center_band(aligned, stat_mode, q_low, q_high):
    if stat_mode == "median":
        center, low, high = compute_median_and_band(aligned, q_low, q_high)
        band_label = f"{q_low}–{q_high} pct"
    else:
        center, sem = compute_mean_and_sem(aligned)
        low, high   = center - sem, center + sem
        band_label  = "SEM"
    return center, low, high, band_label


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


def plot_dpca_components_with_band(res, by_label, component_key="Z_cond",
                                    explained_key="explained_cond", label_key="labels",
                                    n_plot=3, title_prefix="cond",
                                    align_label="target_onset",
                                    q_low=25, q_high=75):
    """
    Like plot_dpca_components but adds per-trial percentile bands.
    by_label: dict[label -> np.ndarray [N, T, H]] from collect_aligned_hidden_by_label.
    """
    rel_time = res["rel_time"]
    Z        = res[component_key]   # [C, T, K]
    labels   = res[label_key]
    pca_key  = "pca_cond" if "cond" in component_key else "pca_time"
    pca      = res[pca_key]

    # grand mean for centering individual trials before projection
    all_h = np.concatenate([by_label[lab] for lab in labels if lab in by_label], axis=0)
    grand = all_h.mean(axis=(0, 1), keepdims=True)   # [1, 1, H]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for k in range(n_plot):
        ax = axes[k]
        for i, lab in enumerate(labels):
            col = colors[i]
            ax.plot(rel_time, Z[i, :, k], color=col, label=str(lab), lw=1.5)
            if lab in by_label:
                h = by_label[lab] - grand           # [N, T, H]
                N, T, H = h.shape
                proj = pca.transform(h.reshape(N * T, H)).reshape(N, T, -1)
                lo = np.percentile(proj[:, :, k], q_low,  axis=0)
                hi = np.percentile(proj[:, :, k], q_high, axis=0)
                ax.fill_between(rel_time, lo, hi, alpha=0.10, color=col)
        ax.axvline(0, linestyle="--")
        ax.set_title(f"{title_prefix}-dPC{k+1}")
        ax.set_xlabel(f"time from {align_label}")
        ax.set_ylabel("component")

    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
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


def plot_jpca_trajectories(jpca_res, labels=None, rel_time=None,
                            align_label="target_onset",
                            cmap="plasma", figsize=(8, 7),
                            title="jPCA rotational dynamics"):
    """jPC1 vs jPC2 phase-plane for each CTOA condition; diamond at t=0."""
    Z = jpca_res["Z"]       # [C, T, 2+]
    C, T, _ = Z.shape

    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, C))

    fig, ax = plt.subplots(figsize=figsize)

    t0_idx = None
    if rel_time is not None:
        hits = np.where(rel_time == 0)[0]
        if len(hits):
            t0_idx = int(hits[0])

    for c in range(C):
        col = colors[c]
        lab = str(labels[c]) if labels is not None else f"cond {c}"

        ax.plot(Z[c, :, 0], Z[c, :, 1], "-", color=col, lw=1.5, label=lab, alpha=0.85)

        ax.scatter(Z[c, 0, 0], Z[c, 0, 1], s=40, color=col,
                   facecolors="none", edgecolors=col, zorder=5, lw=1.5)

        if T >= 2:
            ax.annotate(
                "", xy=(Z[c, -1, 0], Z[c, -1, 1]),
                xytext=(Z[c, -2, 0], Z[c, -2, 1]),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            )

        if t0_idx is not None:
            ax.scatter(Z[c, t0_idx, 0], Z[c, t0_idx, 1],
                       s=70, marker="D", color=col, zorder=6,
                       edgecolors="k", linewidths=0.5)

    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, ls="--", alpha=0.3)
    ax.set_xlabel("jPC1")
    ax.set_ylabel("jPC2")

    var_ratio = jpca_res.get("var_ratio", float("nan"))
    rot_index = jpca_res.get("rot_index", float("nan"))
    ax.set_title(
        f"{title}\n"
        f"rotational var ratio = {var_ratio:.3f}   rot index = {rot_index:.3f}"
    )
    ax.legend(fontsize=7, ncol=2, title="CTOA bin")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig, ax


def plot_jpca_timecourses(jpca_res, labels=None, rel_time=None,
                           align_label="target_onset",
                           cmap="plasma", figsize=(12, 4)):
    """
    jPC1 and jPC2 time courses for each CTOA condition (analogous to dPCA time courses).
    """
    Z = jpca_res["Z"]       # [C, T, 2+]
    C, T, _ = Z.shape
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, C))
    rt = rel_time if rel_time is not None else np.arange(T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for c in range(C):
        col = colors[c]
        lab = str(labels[c]) if labels is not None else f"cond {c}"
        ax1.plot(rt, Z[c, :, 0], color=col, lw=1.5, label=lab)
        ax2.plot(rt, Z[c, :, 1], color=col, lw=1.5, label=lab)

    for ax, name in [(ax1, "jPC1"), (ax2, "jPC2")]:
        ax.axvline(0, ls="--", color="k", lw=0.8, alpha=0.6)
        ax.axhline(0, ls="--", color="k", lw=0.4, alpha=0.3)
        ax.set_xlabel(f"timesteps from {align_label}")
        ax.set_ylabel(name)
        ax.set_title(name)

    ax1.legend(fontsize=7, ncol=2, title="CTOA bin")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_jpca_permutation_test(perm_res, figsize=(10, 4)):
    """Permutation null vs observed for rotational variance ratio and rotational index."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    specs = [
        (axes[0], "perm_var_ratios",  "real_var_ratio",  "p_var_ratio",
         "Rotational variance ratio"),
        (axes[1], "perm_rot_indices", "real_rot_index",  "p_rot_index",
         "Rotational index (||M_skew|| / ||M_opt||)"),
    ]

    for ax, key_perm, key_real, key_p, xlabel in specs:
        nulls = perm_res[key_perm]
        real  = perm_res[key_real]
        p_val = perm_res[key_p]

        ax.hist(nulls, bins=20, color="steelblue", alpha=0.7, label="permutation null")
        ax.axvline(real, color="crimson", lw=2,
                   label=f"observed = {real:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(f"{xlabel}\np = {p_val:.3f}  (n={len(nulls)} perms)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_tangling_timecourses(tang_pre, tang_post=None, dt=20.0, figsize=(8, 5),
                               align_label="target onset"):
    """
    Q(t) per CTOA bin.

    Pass both tang_pre and tang_post to get a single stitched time course
    (pre-target window + post-target window joined at t=0), exactly like
    the paper figure. If tang_post is None, only the pre-target window is shown.

    Color: light cyan (early CTOA) → dark navy (late CTOA), matching paper Blues.
    Background forced white regardless of notebook theme.
    """
    if tang_post is not None:
        # Pre: drop the last step (t=0) to avoid duplicate at join point
        Q_pre    = tang_pre["Q"][:, :-1]
        t_pre    = np.array(tang_pre["rel_time"][:-1]) * dt
        Q_post   = tang_post["Q"]
        t_post   = np.array(tang_post["rel_time"]) * dt
        Q        = np.concatenate([Q_pre, Q_post], axis=1)
        rel_time = np.concatenate([t_pre, t_post])
        labels   = tang_pre["labels"]
    else:
        Q        = tang_pre["Q"]
        rel_time = np.array(tang_pre["rel_time"]) * dt
        labels   = tang_pre["labels"]

    C = Q.shape[0]

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, C))

    _WHITE = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black",   "axes.labelcolor": "black",
        "text.color": "black",       "xtick.color": "black",
        "ytick.color": "black",      "savefig.facecolor": "white",
    }
    with plt.rc_context(_WHITE):
        fig, ax = plt.subplots(figsize=figsize)

        for c in range(C):
            ax.plot(rel_time, Q[c], color=colors[c], lw=2.0)

        ax.axvspan(-30, 30, color="lightgray", alpha=0.45, zorder=0)
        ax.axvline(0, color="gray", lw=1.0, ls="--", alpha=0.6)

        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Tangling", fontsize=12)
        ax.set_title("Trajectory tangling per CTOA bin", fontsize=13)
        ax.set_xlim(rel_time[0], rel_time[-1])
        ax.set_ylim(bottom=0)

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma,
            norm=plt.Normalize(vmin=0, vmax=C - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("CTOA bin", fontsize=9)
        cbar.set_ticks([0, C - 1])
        cbar.set_ticklabels(["early", "late"])
        cbar.ax.yaxis.set_tick_params(labelsize=9)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=11, colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")

        plt.tight_layout()
        plt.show()

    plt.close(fig)
    return fig


def _sig_stars(p):
    """Return significance stars string matching paper convention."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def plot_tangling_vs_ctoa(tang_pre, tang_post, figsize=(7, 5)):
    """
    Scatter: mean tangling vs CTOA (ms).

    Both pre-target (■ squares) and post-target (● circles) on the same axes
    with dual y-axes (scales differ by ~50×). Points colored by CTOA bin
    (plasma colormap). Dashed regression lines: red = pre, magenta = post.
    """
    x_pre  = tang_pre["ctoa_ms_mean"]
    y_pre  = tang_pre["Q_mean"]
    x_post = tang_post["ctoa_ms_mean"]
    y_post = tang_post["Q_mean"]
    n      = len(x_pre)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n))

    _WHITE = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black",   "axes.labelcolor": "black",
        "text.color": "black",       "xtick.color": "black",
        "ytick.color": "black",      "savefig.facecolor": "white",
    }
    with plt.rc_context(_WHITE):
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        for i in range(n):
            ax1.scatter(x_pre[i], y_pre[i], marker="s", s=70,
                        color=colors[i], edgecolors="black",
                        linewidths=0.5, zorder=5)

        reg_pre = polynomial_regression(x_pre, y_pre, degree=2)
        if reg_pre["coeffs"] is not None:
            x_line = np.linspace(np.nanmin(x_pre), np.nanmax(x_pre), 200)
            y_line = np.polyval(reg_pre["coeffs"], x_line)
            ax1.plot(x_line, y_line, color="red", ls="--", lw=1.8)
            stars = _sig_stars(reg_pre["p_value"])
            print(f"Pre-target:  R²={reg_pre['r2']:.3f}  "
                  f"p={reg_pre['p_value']:.4f}  (deg 2)")

        ax1.set_ylabel("Tangling (pre target)", fontsize=11)
        ax1.tick_params(axis="y", labelsize=10)

        for i in range(n):
            ax2.scatter(x_post[i], y_post[i], marker="o", s=70,
                        color=colors[i], edgecolors="black",
                        linewidths=0.5, zorder=5)

        reg_post = polynomial_regression(x_post, y_post, degree=1)
        if reg_post["coeffs"] is not None:
            x_line = np.linspace(np.nanmin(x_post), np.nanmax(x_post), 200)
            y_line = np.polyval(reg_post["coeffs"], x_line)
            ax2.plot(x_line, y_line, color="magenta", ls="--", lw=1.8)
            stars_post = _sig_stars(reg_post["p_value"])
            print(f"Post-target: R²={reg_post['r2']:.3f}  "
                  f"p={reg_post['p_value']:.4f}  (deg 1)")

        ax2.set_ylabel("Tangling (post target)", fontsize=11)
        ax2.tick_params(axis="y", labelsize=10)

        from matplotlib.lines import Line2D
        legend_handles = []
        if reg_pre["coeffs"] is not None:
            stars = _sig_stars(reg_pre["p_value"])
            legend_handles.append(Line2D(
                [0], [0], marker="s", linestyle="none",
                markerfacecolor="gray", markeredgecolor="black", markeredgewidth=0.5,
                markersize=8, label=f"Pre Target   R² = {reg_pre['r2']:.2f} {stars}",
            ))
        if reg_post["coeffs"] is not None:
            stars_post = _sig_stars(reg_post["p_value"])
            legend_handles.append(Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="gray", markeredgecolor="black", markeredgewidth=0.5,
                markersize=8, label=f"Post Target  R² = {reg_post['r2']:.2f} {stars_post}",
            ))
        if legend_handles:
            ax1.legend(handles=legend_handles, fontsize=9,
                       loc="upper center", bbox_to_anchor=(0.5, -0.18),
                       ncol=2, framealpha=0.85, edgecolor="lightgray")

        ax1.set_xlabel("CTOA (ms)", fontsize=11)
        ax1.set_title("Tangling vs CTOA", fontsize=12)

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma,
            norm=plt.Normalize(vmin=0, vmax=n - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, pad=0.14)
        cbar.set_label("CTOA bin", fontsize=9)
        cbar.set_ticks([0, n - 1])
        cbar.set_ticklabels(["early", "late"])

        for spine in ax1.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
        ax1.tick_params(axis="x", labelsize=10, colors="black")

        plt.tight_layout()
        plt.show()

    plt.close(fig)
    return fig


def plot_tangling_vs_position_info(tang_pre, decoding_acc_pre, figsize=(6, 5)):
    """Pre-target tangling vs position decoding accuracy per CTOA bin (paper: rho~0.7)."""
    from scipy.stats import spearmanr

    x = decoding_acc_pre
    y = tang_pre["Q_mean"]
    mask = np.isfinite(x) & np.isfinite(y)

    rho, p = spearmanr(x[mask], y[mask])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=60, zorder=5, color="steelblue")
    for i, lab in enumerate(tang_pre["labels"]):
        ax.annotate(str(lab), (x[i], y[i]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("position decoding accuracy (pre-target)")
    ax.set_ylabel("mean tangling Q (pre-target)")
    ax.set_title(f"Tangling vs position info (pre-target)\n"
                 f"Spearman rho={rho:.2f}  p={p:.3f}   (paper: rho~0.7)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print(f"Spearman rho={rho:.3f}  p={p:.4f}")
    return fig, rho, p


def plot_tangling_vs_rt(tang_post, rt_by_bin, figsize=(5, 5)):
    """
    Scatter: post-target tangling vs mean RT per CTOA bin.
    Filled circles colored by CTOA bin (Blues gradient), ρ annotation.
    """
    from scipy.stats import spearmanr

    labels = tang_post["labels"]
    y      = tang_post["Q_mean"]
    x      = np.array([rt_by_bin.get(b, np.nan) for b in labels])
    mask   = np.isfinite(x) & np.isfinite(y)

    rho, p = spearmanr(x[mask], y[mask])

    n      = len(labels)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n))

    _WHITE = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black",   "axes.labelcolor": "black",
        "text.color": "black",       "xtick.color": "black",
        "ytick.color": "black",      "savefig.facecolor": "white",
    }
    with plt.rc_context(_WHITE):
        fig, ax = plt.subplots(figsize=figsize)

        for i in range(n):
            ax.scatter(x[i], y[i], s=65, color=colors[i],
                       edgecolors="black", linewidths=0.5, zorder=5)

        stars = _sig_stars(p)
        ax.text(0.97, 0.97,
                f"ρ = {rho:.2f} {stars}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontstyle="italic")

        ax.set_xlabel("Reaction time (ms)", fontsize=11)
        ax.set_ylabel("Tangling post target", fontsize=11)
        ax.set_title("Tangling vs RT (post-target)", fontsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()

    plt.close(fig)
    print(f"Spearman rho={rho:.3f}  p={p:.4f}")
    return fig, rho, p


def plot_decoding_vs_ctoa(dec_pre, dec_post, figsize=(13, 5)):
    """
    Two-panel plot: decoding accuracy vs CTOA for pre- and post-target windows.
    Fits polynomial regressions (linear + quadratic) and reports AIC comparison.

    Paper results:
      Pre-target:  R²=0.04  (not significant)
      Post-target: R²=0.64  quadratic  (p significant)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    specs = [
        (axes[0], dec_pre,  "Pre-target decoding vs CTOA"),
        (axes[1], dec_post, "Post-target decoding vs CTOA"),
    ]

    for ax, dec, title in specs:
        if dec is None:
            ax.set_title(f"{title}\n(no data)")
            continue

        x   = dec["ctoa_ms_mean"]
        y   = dec["acc_per_bin"]
        sem = dec["sem_per_bin"]

        ax.axhline(dec["chance"], color="gray", ls="--", lw=1, alpha=0.6,
                   label=f"chance ({dec['chance']:.2f})")
        ax.errorbar(x, y, yerr=sem, fmt="o", color="steelblue",
                    ms=7, capsize=4, zorder=5)

        for deg, col, ls in [(1, "steelblue", "-"), (2, "crimson", "--")]:
            reg = polynomial_regression(x, y, degree=deg)
            if reg["coeffs"] is not None:
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                y_line = np.polyval(reg["coeffs"], x_line)
                n = len(x)
                k = deg + 1
                ss_res = np.sum((reg["y"] - reg["y_hat"]) ** 2)
                aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.nan
                ax.plot(x_line, y_line, color=col, ls=ls, lw=1.8,
                        label=f"deg-{deg}  R²={reg['r2']:.2f}  p={reg['p_value']:.3f}  AIC={aic:.1f}")
                print(f"{title.split(chr(10))[0]} deg-{deg}: "
                      f"R²={reg['r2']:.3f}  p={reg['p_value']:.4f}  AIC={aic:.1f}")

        ax.set_xlabel("mean CTOA (ms)")
        ax.set_ylabel("decoding accuracy")
        ax.set_title(title)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_decoding_timecourse(trials, align_key="target_onset",
                              window_step_ms=20, window_width_ms=100,
                              ctoa_bin_groups=None, dt=20,
                              outcome="correct", figsize=(10, 5),
                              pca_dims=None):
    """
    Sliding-window decoding accuracy over time, split by early/late CTOA.

    Shows when during the trial the network's hidden state carries
    spatial information about the target location.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from collections import defaultdict

    if ctoa_bin_groups is None:
        ctoa_bin_groups = {
            "early CTOA (bins 0-2)": list(range(0, 3)),
            "late  CTOA (bins 7-9)": list(range(7, 10)),
        }

    # Find global time range
    filtered = [tr for tr in trials
                if outcome is None or tr.get("train_outcome") == outcome]
    if not filtered:
        print("No matching trials.")
        return

    # Align all trials to target_onset; find common pre/post range
    max_pre  = min(tr[align_key] for tr in filtered if tr.get(align_key))
    max_post = min(tr["h"].shape[0] - tr[align_key]
                   for tr in filtered if tr.get(align_key))
    max_pre  = min(max_pre, int(round(1500 / dt)))   # cap at 1500ms
    max_post = min(max_post, int(round(800 / dt)))    # cap at 800ms

    step   = max(1, int(round(window_step_ms / dt)))
    half_w = max(1, int(round(window_width_ms / dt)) // 2)
    t_centers = np.arange(-max_pre + half_w, max_post - half_w, step)

    colors = ["steelblue", "crimson", "forestgreen", "orange"]
    fig, ax = plt.subplots(figsize=figsize)

    for (group_label, bins), col in zip(ctoa_bin_groups.items(), colors):
        group_trials = [tr for tr in filtered if tr.get("ctoa_bin") in bins]
        if len(group_trials) < 20:
            continue

        acc_tc = []
        for tc in t_centers:
            t0_steps = int(tc)
            feats, locs = [], []
            for tr in group_trials:
                t0 = tr[align_key]
                a  = t0 + t0_steps - half_w
                b  = t0 + t0_steps + half_w
                if a < 0 or b > tr["h"].shape[0]:
                    continue
                feats.append(tr["h"][a:b].mean(axis=0))
                locs.append(tr["target_loc"] - 1)

            if len(feats) < 20:
                acc_tc.append(np.nan)
                continue

            X = np.stack(feats)
            y = np.array(locs)
            min_cls = min(np.bincount(y, minlength=4))
            if min_cls < 2:
                acc_tc.append(np.nan)
                continue

            if pca_dims is not None and pca_dims < X.shape[1]:
                pca = PCA(n_components=pca_dims)
                X = pca.fit_transform(X)

            scaler = StandardScaler()
            clf    = LinearDiscriminantAnalysis()
            cv     = StratifiedKFold(n_splits=min(5, min_cls), shuffle=True,
                                      random_state=0)
            fold_accs = []
            for tr_i, te_i in cv.split(X, y):
                clf.fit(scaler.fit_transform(X[tr_i]), y[tr_i])
                fold_accs.append(clf.score(scaler.transform(X[te_i]), y[te_i]))
            acc_tc.append(np.mean(fold_accs))

        t_ms = t_centers * dt
        acc_tc = np.array(acc_tc)
        ax.plot(t_ms, acc_tc, color=col, lw=2,
                label=f"{group_label} (n={len(group_trials)})")

    ax.axhline(0.25, color="gray", ls="--", lw=1, alpha=0.6, label="chance")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label=align_key)
    ax.set_xlabel(f"time from {align_key} (ms)")
    ax.set_ylabel("decoding accuracy (location)")
    ax.set_title("Position decoding timecourse\nearly vs late CTOA")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_decoding_vs_ctoa_overlay(dec_pre, dec_post, figsize=(5.5, 5),
                                  as_percent=True):
    """Single-panel decoding accuracy vs CTOA.

    Both windows on one axis so they're directly comparable: pre-target = circles,
    post-target = squares, points colored by CTOA (light=early -> dark=late) on a
    Blues gradient. One dashed quadratic fit per series with an R² + significance
    label. No titles or per-fit legend clutter.
    """
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    scale = 100.0 if as_percent else 1.0
    series = [
        ("Pre Target",  dec_pre,  "o"),
        ("Post Target", dec_post, "s"),
    ]
    n_bins = max((len(d["acc_per_bin"]) for _, d, _ in series if d is not None),
                 default=10)
    cmap = plt.cm.Blues
    norm = Normalize(vmin=-0.15 * n_bins, vmax=n_bins - 1)  # avoid near-white start

    _WHITE = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black",
        "ytick.color": "black", "savefig.facecolor": "white",
    }
    with plt.rc_context(_WHITE):
        fig, ax = plt.subplots(figsize=figsize)

        r2_labels = []  # (series label, marker, R² text) stacked in a corner so they never overlap
        for label, dec, marker in series:
            if dec is None:
                continue
            x = dec["ctoa_ms_mean"]
            y = dec["acc_per_bin"] * scale
            sem = dec["sem_per_bin"] * scale
            colors = cmap(norm(np.arange(len(x))))

            ax.errorbar(x, y, yerr=sem, fmt="none", ecolor="0.6",
                        elinewidth=1.0, capsize=2.5, zorder=2)
            ax.scatter(x, y, marker=marker, s=55, c=colors,
                       edgecolors="black", linewidths=0.5, zorder=4)

            reg = polynomial_regression(x, y, degree=2)
            if reg["coeffs"] is not None:
                xl = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ax.plot(xl, np.polyval(reg["coeffs"], xl), "--",
                        color="black", lw=1.3, zorder=3)
                r2_labels.append(f"{label}:  R² = {reg['r2']:.2f} {_sig_stars(reg['p_value'])}")

        for i, txt in enumerate(r2_labels):
            ax.text(0.97, 0.06 + 0.07 * (len(r2_labels) - 1 - i), txt,
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, fontstyle="italic")

        ax.set_xlabel("CTOA (ms)", fontsize=12)
        ax.set_ylabel("% Accuracy" if as_percent else "Decoding accuracy",
                      fontsize=12)
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.4",
                   markeredgecolor="black", markersize=8, label="Pre Target"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="0.4",
                   markeredgecolor="black", markersize=8, label="Post Target"),
        ]
        ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
                  frameon=False)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                            fraction=0.05, pad=0.14, aspect=30)
        cbar.set_ticks([0, n_bins - 1])
        cbar.set_ticklabels(["CTOA early", "CTOA late"])
        cbar.ax.tick_params(length=0, labelsize=9)

        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return fig
