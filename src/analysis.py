"""
Post-hoc analysis: trial collection, PCA, dPCA, spatial separation.

Trial dict keys (set by collect_trials):
    x, gt, h, logits, a
    cue_onset, target_onset, target_win_on, target_win_off, model_resp_on
    cue_loc, target_loc, ctoa_ms, ctoa_bin
    has_distractors, n_distractors, distractor_locs, distractor_cdoa_ms
    first_distractor_cdoa_ms
    train_outcome, event_type, rt_ms
"""

import math
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ collect

@torch.no_grad()
def collect_trials(model, env_fn, n_trials=200, device="cpu"):
    """
    Roll out n_trials with greedy (argmax) policy.
    Outcome reconstruction uses the env's distractor event list directly
    (more accurate than just checking gt_seq[t]).
    """
    model.eval()
    dev = torch.device(device)
    trials = []

    for _ in range(n_trials):
        env = env_fn()
        env.reset()

        x_seq  = env.ob.copy().astype(np.float32)
        gt_seq = env.gt.copy().astype(np.int64)

        x_t = torch.tensor(x_seq, dtype=torch.float32, device=dev).unsqueeze(0)
        logits, _, h_seq = model(x_t, return_hidden=True)

        logits_np = logits[0].cpu().numpy()       # [T, 2]
        h_np      = h_seq[0].cpu().numpy()        # [T, H]
        actions   = np.argmax(logits_np, axis=-1) # [T]

        cue_onset    = int(env.start_ind["cue"])
        target_onset = int(env.start_ind["target"])

        win_inds     = np.where(gt_seq == 1)[0]
        target_win_on  = int(win_inds[0])       if len(win_inds) > 0 else None
        target_win_off = int(win_inds[-1] + 1)  if len(win_inds) > 0 else None

        first_release  = np.where(actions == 1)[0]
        model_resp_on  = int(first_release[0])  if len(first_release) > 0 else None

        if model_resp_on is None:
            train_outcome = "miss"
            event_type    = "target"
            rt_ms         = None
        else:
            t = model_resp_on
            if gt_seq[t] == 1:
                train_outcome = "correct"
                event_type    = "target"
                rt_ms         = int((t - target_onset) * env.dt)
            else:
                active_fa = None
                for ev in env._distractor_events:
                    if ev["fa_start"] <= t < ev["fa_end"]:
                        active_fa = ev
                        break
                if active_fa is not None:
                    train_outcome = "false_alarm"
                    event_type    = "distractor"
                    rt_ms         = int((t - active_fa["onset_step"]) * env.dt)
                else:
                    train_outcome = "abort"
                    event_type    = "none"
                    rt_ms         = None

        trials.append({
            "x":       x_seq,
            "gt":      gt_seq,
            "h":       h_np,
            "logits":  logits_np,
            "a":       actions,

            "cue_onset":     cue_onset,
            "target_onset":  target_onset,
            "target_win_on": target_win_on,
            "target_win_off":target_win_off,
            "model_resp_on": model_resp_on,

            "cue_loc":     env.trial.get("cue_loc"),
            "target_loc":  env.trial.get("target_loc"),
            "ctoa_ms":     env.trial.get("ctoa_ms"),
            "ctoa_bin":    env.trial.get("ctoa_bin"),
            "has_distractors":        env.trial.get("has_distractors", False),
            "n_distractors":          env.trial.get("n_distractors", 0),
            "distractor_locs":        env.trial.get("distractor_locs", []),
            "distractor_cdoa_ms":     env.trial.get("distractor_cdoa_ms", []),
            "first_distractor_cdoa_ms": env.trial.get("first_distractor_cdoa_ms"),

            "train_outcome": train_outcome,
            "event_type":    event_type,
            "rt_ms":         rt_ms,
        })

    return trials


# ------------------------------------------------------------------ filtering

def filter_trials(trials, outcome=None, require_distractors=None, target_locs=None):
    """Simple trial subset selector. All args are optional / combinable."""
    kept = []
    for tr in trials:
        if outcome is not None and tr.get("train_outcome") != outcome:
            continue
        if require_distractors is not None and tr.get("has_distractors") != require_distractors:
            continue
        if target_locs is not None and tr.get("target_loc") not in target_locs:
            continue
        kept.append(tr)
    return kept


def select_trials(trials, trial_proj=None, train_outcome=None, has_distractors=None,
                  ctoa_bin_min=None, ctoa_bin_max=None, target_loc=None):
    """
    Filter by metadata. Returns (trials, proj) if trial_proj given, else just trials.
    Accepts both old 'train_outcome' kwarg and the filter_trials-style 'outcome'.
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


# ------------------------------------------------------------------ PCA on hidden states

def fit_pca_on_trials(trials, n_components=3):
    """
    Fit PCA on concatenated hidden states, project per trial.
    Returns pca, trial_proj list, explained_variance_ratio.
    """
    all_h = np.concatenate([tr["h"] for tr in trials], axis=0)
    pca = PCA(n_components=n_components)
    Z   = pca.fit_transform(all_h)

    trial_proj, start = [], 0
    for tr in trials:
        T = tr["h"].shape[0]
        trial_proj.append(Z[start:start + T])
        start += T

    return pca, trial_proj, pca.explained_variance_ratio_


# ------------------------------------------------------------------ aligned windows

def get_aligned_pca_segments(
    trials, trial_proj,
    align_key="target_onset",
    window_before=40, window_after=40,
    train_outcome=None, has_distractors=None,
    ctoa_bin_min=None, ctoa_bin_max=None, target_loc=None,
):
    """
    Cut event-aligned windows from PCA projections.
    Returns aligned [N, W, D], rel_time [W], kept_trials list.
    (returns None, None, [] if nothing matched)
    """
    filtered, filtered_proj = select_trials(
        trials, trial_proj=trial_proj,
        train_outcome=train_outcome, has_distractors=has_distractors,
        ctoa_bin_min=ctoa_bin_min, ctoa_bin_max=ctoa_bin_max, target_loc=target_loc,
    )

    segs, kept = [], []
    for tr, proj in zip(filtered, filtered_proj):
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        s, e = t0 - window_before, t0 + window_after + 1
        if s < 0 or e > len(proj):
            continue
        segs.append(proj[s:e])
        kept.append(tr)

    if not segs:
        return None, None, []

    return np.stack(segs, axis=0), np.arange(-window_before, window_after + 1), kept


# ------------------------------------------------------------------ stats

def compute_median_and_band(aligned, q_low=25, q_high=75):
    """aligned [N,W,D] -> median, low, high each [W,D]"""
    return (np.median(aligned, axis=0),
            np.percentile(aligned, q_low,  axis=0),
            np.percentile(aligned, q_high, axis=0))


def compute_mean_and_sem(aligned):
    """aligned [N,W,D] -> mean, sem each [W,D]"""
    mean = np.mean(aligned, axis=0)
    n    = aligned.shape[0]
    sem  = np.std(aligned, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


# ------------------------------------------------------------------ dPCA (marginal style)

def dpca_marginals(X, n_components=3):
    """
    Approximate dPCA via separate PCAs on time- and condition-marginals of X [C, T, H].

    Returns dict with Z_time, Z_cond [C,T,K], pca_time, pca_cond,
    explained_time, explained_cond, X_time, X_cond.
    """
    C, T, H = X.shape
    grand  = X.mean(axis=(0, 1), keepdims=True)
    X_time = np.repeat(X.mean(axis=0, keepdims=True) - grand, C, axis=0)
    X_cond = X - grand - X_time

    pca_time = PCA(n_components=n_components)
    Z_time   = pca_time.fit_transform(X_time.reshape(C * T, H)).reshape(C, T, n_components)

    pca_cond = PCA(n_components=n_components)
    Z_cond   = pca_cond.fit_transform(X_cond.reshape(C * T, H)).reshape(C, T, n_components)

    return {
        "X_time": X_time, "X_cond": X_cond,
        "pca_time": pca_time, "pca_cond": pca_cond,
        "Z_time": Z_time, "Z_cond": Z_cond,
        "explained_time": pca_time.explained_variance_ratio_,
        "explained_cond": pca_cond.explained_variance_ratio_,
    }


def collect_aligned_hidden_by_label(trials, label_fn, align_key="target_onset",
                                    window_before=40, window_after=40):
    """
    Group event-aligned hidden-state windows by label_fn(trial).
    Returns by_label dict[label -> [N, W, H]], rel_time [W].
    """
    by_label = {}
    rel_time = np.arange(-window_before, window_after + 1)

    for tr in trials:
        label = label_fn(tr)
        if label is None:
            continue
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        h = tr["h"]
        s, e = t0 - window_before, t0 + window_after + 1
        if s < 0 or e > len(h):
            continue
        by_label.setdefault(label, []).append(h[s:e])

    return {k: np.stack(v, axis=0) for k, v in by_label.items()}, rel_time


def make_condition_mean_tensor(by_label, min_trials=3):
    """
    Average within conditions. Returns X [C,W,H], labels list, counts list.
    (X is None if nothing survived min_trials)
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


# ------------------------------------------------------------------ spatial analysis

def extract_window_features(trials, key="h", align_key="target_onset",
                             window_ms=(-300, 0), dt=20):
    """
    Average activity in a time window relative to align_key, one vector per trial.

    Returns X [N, D], y_loc [N] (0-indexed, 0..3), keep_idx list[int].
    """
    feats, y_loc, keep_idx = [], [], []

    for i, tr in enumerate(trials):
        t0 = tr.get(align_key)
        if t0 is None:
            continue

        arr = tr[key]
        T   = arr.shape[0]
        a   = max(0, t0 + int(round(window_ms[0] / dt)))
        b   = min(T, t0 + int(round(window_ms[1] / dt)))
        if b <= a:
            continue

        feats.append(arr[a:b].mean(axis=0))
        y_loc.append(tr["target_loc"] - 1)   # 1..4 -> 0..3
        keep_idx.append(i)

    return np.stack(feats, axis=0), np.array(y_loc, dtype=np.int64), keep_idx


def plot_spatial_separation_pca(trials, key="h", align_key="target_onset",
                                 window_ms=(-300, 0), dt=20, standardize=True,
                                 annotate_centroids=True, alpha=0.65,
                                 s_points=35, s_centroids=180):
    """
    PCA scatter of trial-averaged activity in window_ms, colored by target location.
    Returns dict with X, X_plot, y_loc, Z, keep_idx, pca, centroids.
    """
    import matplotlib.pyplot as plt

    X, y_loc, keep_idx = extract_window_features(trials, key, align_key, window_ms, dt)

    X_plot = StandardScaler().fit_transform(X) if standardize else X

    pca = PCA(n_components=2, random_state=0)
    Z   = pca.fit_transform(X_plot)

    loc_names = ["loc1", "loc2", "loc3", "loc4"]
    centroids = []

    plt.figure(figsize=(7, 6))
    for loc in range(4):
        m = (y_loc == loc)
        if not np.any(m):
            centroids.append(np.array([np.nan, np.nan]))
            continue
        plt.scatter(Z[m, 0], Z[m, 1], s=s_points, alpha=alpha, label=loc_names[loc])
        c = Z[m].mean(axis=0)
        centroids.append(c)
        plt.scatter(c[0], c[1], s=s_centroids, marker="X")
        if annotate_centroids:
            plt.text(c[0], c[1], loc_names[loc], fontsize=10)

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({100*evr[0]:.1f}% var)")
    plt.ylabel(f"PC2 ({100*evr[1]:.1f}% var)")
    plt.title(f"{key} | window {window_ms[0]}..{window_ms[1]} ms re {align_key}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "X": X, "X_plot": X_plot, "y_loc": y_loc,
        "Z": Z, "keep_idx": keep_idx,
        "pca": pca, "centroids": np.stack(centroids, axis=0),
    }


def centroid_distance_matrix(centroids):
    """centroids [4, 2] -> D [4, 4] pairwise Euclidean distances"""
    n = len(centroids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    return D


def print_centroid_distances(res, name=""):
    D = centroid_distance_matrix(res["centroids"])
    if name:
        print(name)
    print(np.round(D, 3))
    return D


def compare_spatial_separation(trials_A, trials_B, label_A="A", label_B="B",
                                align_key="target_onset", dt=20):
    """
    Pre- and post-target spatial separation for two trial subsets side by side.
    Returns nested dict A/B -> pre/post -> plot_spatial_separation_pca result.
    """
    print(f"{label_A}: n={len(trials_A)},  {label_B}: n={len(trials_B)}")
    out = {}

    for trials, label, key in [(trials_A, label_A, "A"), (trials_B, label_B, "B")]:
        print(f"\n--- {label} pre-target ---")
        pre  = plot_spatial_separation_pca(trials, align_key=align_key,
                                           window_ms=(-300, 0), dt=dt)
        print_centroid_distances(pre, name=f"{label} pre")

        print(f"\n--- {label} post-target ---")
        post = plot_spatial_separation_pca(trials, align_key=align_key,
                                           window_ms=(100, 300), dt=dt)
        print_centroid_distances(post, name=f"{label} post")

        out[key] = {"pre": pre, "post": post}

    return out
