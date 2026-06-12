"""
Post-hoc analysis: trial collection, PCA, dPCA, jPCA, spatial separation.

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
from collections import defaultdict
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
            # Continuous-mode geometric fields (also populated in legacy
            # discrete mode, where they equal CUE_POS[loc] / STIM_POS[loc]).
            "cue_pos":      env.trial.get("cue_pos"),
            "target_pos":   env.trial.get("target_pos"),
            "cue_theta":    env.trial.get("cue_theta"),
            "target_theta": env.trial.get("target_theta"),
            "distractor_positions": env.trial.get("distractor_positions", []),
            "distractor_thetas":    env.trial.get("distractor_thetas", []),
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


@torch.no_grad()
def collect_trials_spatial(model, env_fn, n_trials=200, device="cpu", head="spatial"):
    """Spatial analog of collect_trials.

    head="spatial" (default): the model output [T, 2+] is interpreted as
    continuous (x, y) (first two dims); each step's (x, y) is fed into the
    spatial env, which computes the position-based outcome. `train_outcome` in
    {"correct", "false_alarm", "miss"}.

    head="go_nogo": the model's W_out is binary go/no-go (the auxiliary (x, y)
    readout is model.decode_xy). Delegates to the binary `collect_trials` for the
    correct go/no-go outcome logic, then attaches `xy_pred = decode_xy(h)` so the
    _spatial analysis notebooks can still read a metric position estimate.
    """
    if head == "go_nogo":
        trials = collect_trials(model, env_fn, n_trials=n_trials, device=device)
        if hasattr(model, "decode_xy"):
            with torch.no_grad():
                for tr in trials:
                    h_t = torch.tensor(tr["h"], dtype=torch.float32, device=device)
                    xy = model.decode_xy(h_t).cpu().numpy()      # [T, 2]
                    tr["xy_pred"] = xy
                    tx, ty = tr.get("target_pos", (0.0, 0.0))
                    tr["mean_err"] = float(np.hypot(xy[:, 0] - tx, xy[:, 1] - ty).mean())
        return trials

    model.eval()
    dev = torch.device(device)
    trials = []

    for _ in range(n_trials):
        env = env_fn()
        env.reset()

        x_seq  = env.ob.copy().astype(np.float32)
        gt_seq = env.gt.copy().astype(np.int64)
        T = x_seq.shape[0]

        x_t = torch.tensor(x_seq, dtype=torch.float32, device=dev).unsqueeze(0)
        out, _, h_seq = model(x_t, return_hidden=True)

        out_np = out[0].cpu().numpy()       # [T, out_dim]
        h_np   = h_seq[0].cpu().numpy()     # [T, H]
        # Multi-task output: first 2 = (x, y) position; remaining (if any) =
        # hold/release logits. Single-head (out_dim==2) checkpoints still work.
        xy_np     = out_np[:, :2]
        logits_np = out_np[:, 2:] if out_np.shape[1] > 2 else out_np

        # Step env with the model's per-timestep (x, y) so it populates
        # _step_actions and the final outcome.
        final_info = None
        for t in range(T):
            _obs, _r, terminated, _trunc, info = env.step(xy_np[t])
            if terminated:
                final_info = info
                break
        # In case the loop ended without terminated=True (shouldn't happen
        # given _step's terminal condition), still query env for partial state.
        if final_info is None:
            final_info = info

        cue_onset    = int(env.start_ind["cue"])
        target_onset = int(env.start_ind["target"])

        # Reaction time from the hold/release TIMING head (out[:, 2:]). The env's
        # position-based rt_ms reflects only when the (x,y) head nears the target,
        # which is held from cue onset and so is ~flat across CTOA. The "when"
        # decision lives in the release head — read RT from it (first release =
        # argmax==1 inside the response window). Falls back to env rt if no head.
        rt_release = final_info.get("rt_ms")
        if logits_np.shape[1] >= 2:
            rw0, rw1 = env.rt_window
            r0 = target_onset + int(round(rw0 / env.dt))
            r1 = target_onset + int(round(rw1 / env.dt))
            released = np.argmax(logits_np, axis=1) == 1
            rt_release = None
            for t in range(max(r0, 0), min(r1, logits_np.shape[0])):
                if released[t]:
                    rt_release = int((t - target_onset) * env.dt)
                    break

        trials.append({
            "x":       x_seq,
            "gt":      gt_seq,
            "h":       h_np,
            "logits":  logits_np,  # hold/release logits (timing head), or (x,y) if single-head
            "xy_pred": xy_np,      # (x, y) position head
            "a":       xy_np,      # per-step (x, y) action fed to the env

            "cue_onset":     cue_onset,
            "target_onset":  target_onset,
            # Binary-task-only fields; kept as None to satisfy any analysis
            # code that does tr.get(...) on them.
            "target_win_on":  None,
            "target_win_off": None,
            "model_resp_on":  None,

            "cue_loc":     env.trial.get("cue_loc"),
            "target_loc":  env.trial.get("target_loc"),
            "cue_pos":      env.trial.get("cue_pos"),
            "target_pos":   env.trial.get("target_pos"),
            "cue_theta":    env.trial.get("cue_theta"),
            "target_theta": env.trial.get("target_theta"),
            "distractor_positions": env.trial.get("distractor_positions", []),
            "distractor_thetas":    env.trial.get("distractor_thetas", []),
            "ctoa_ms":     env.trial.get("ctoa_ms"),
            "ctoa_bin":    env.trial.get("ctoa_bin"),
            "has_distractors":        env.trial.get("has_distractors", False),
            "n_distractors":          env.trial.get("n_distractors", 0),
            "distractor_locs":        env.trial.get("distractor_locs", []),
            "distractor_cdoa_ms":     env.trial.get("distractor_cdoa_ms", []),
            "first_distractor_cdoa_ms": env.trial.get("first_distractor_cdoa_ms"),

            "train_outcome": final_info["train_outcome"],
            "mean_err":      final_info.get("mean_err"),
            "distractor_pull": final_info.get("distractor_pull"),
            "rt_ms":         rt_release,                    # release-head RT (timing)
            "rt_ms_pos":     final_info.get("rt_ms"),       # position-based RT (legacy)
            "event_type":    "target",  # placeholder; spatial task has no "type"
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


# ------------------------------------------------------------------ linear decoder

def decode_position_by_ctoa_bin(trials, window_ms=(-300, 0), align_key="target_onset",
                                  dt=20, outcome="correct", min_trials=10,
                                  n_splits=5, shuffle_seed=0, pca_dims=None):
    """
    LDA decoding of target location from mean hidden state in window_ms,
    evaluated per CTOA bin via stratified k-fold CV.

    pca_dims: if set, pre-reduces features with PCA before LDA (helps when
    position is encoded in a high-dimensional subspace).
    Returns dict: acc_per_bin, sem_per_bin, labels, ctoa_ms_mean, chance, n_per_bin.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    by_bin = defaultdict(list)
    for tr in trials:
        if outcome is not None and tr.get("train_outcome") != outcome:
            continue
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        h = tr["h"]
        T = h.shape[0]
        a = max(0, t0 + int(round(window_ms[0] / dt)))
        b = min(T, t0 + int(round(window_ms[1] / dt)))
        if b <= a:
            continue
        b_idx = tr.get("ctoa_bin")
        if b_idx is None:
            continue
        feat = h[a:b].mean(axis=0)   # [N_neurons]
        loc  = tr["target_loc"] - 1  # 0..3
        by_bin[b_idx].append((feat, loc))

    if pca_dims is not None:
        all_feats = np.stack([f for items in by_bin.values() for f, _ in items])
        pca = PCA(n_components=min(pca_dims, all_feats.shape[1]))
        pca.fit(all_feats)
        by_bin_proj = defaultdict(list)
        for b, items in by_bin.items():
            for feat, loc in items:
                by_bin_proj[b].append((pca.transform(feat[None])[0], loc))
        by_bin = by_bin_proj

    labels, acc_list, sem_list, ctoa_ms_list, n_list = [], [], [], [], []
    bin_to_ms = defaultdict(list)
    for tr in trials:
        b = tr.get("ctoa_bin")
        ms = tr.get("ctoa_ms")
        if b is not None and ms is not None:
            bin_to_ms[b].append(ms)

    for b in sorted(by_bin.keys()):
        items = by_bin[b]
        if len(items) < min_trials:
            continue

        X = np.stack([x for x, _ in items], axis=0)   # [N, D]
        y = np.array([l for _, l in items], dtype=np.int64)

        # Need at least n_splits examples per class for stratified CV
        min_per_class = min(np.bincount(y, minlength=4))
        actual_splits = min(n_splits, int(min_per_class))
        if actual_splits < 2:
            continue

        scaler = StandardScaler()
        clf    = LinearDiscriminantAnalysis()
        cv     = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                                  random_state=shuffle_seed)

        fold_accs = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf.fit(X_tr, y[train_idx])
            fold_accs.append(clf.score(X_te, y[test_idx]))

        labels.append(b)
        acc_list.append(np.mean(fold_accs))
        sem_list.append(np.std(fold_accs) / np.sqrt(len(fold_accs)) if len(fold_accs) > 1 else 0.0)
        ctoa_ms_list.append(np.mean(bin_to_ms[b]) if b in bin_to_ms else np.nan)
        n_list.append(len(items))

    return {
        "acc_per_bin":  np.array(acc_list),
        "sem_per_bin":  np.array(sem_list),
        "labels":       labels,
        "ctoa_ms_mean": np.array(ctoa_ms_list),
        "chance":       0.25,
        "n_per_bin":    n_list,
        "window_ms":    window_ms,
        "align_key":    align_key,
        "pca_dims":     pca_dims,
    }


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


# ------------------------------------------------------------------ jPCA

def prepare_jpca_input(trials, align_key="target_onset",
                        window_before=15, window_after=30,
                        min_trials=5, outcome="correct"):
    """
    Average hidden states per CTOA bin to build X_cond [C, T, N].
    Returns X_cond, labels, rel_time, counts (or None if no valid bins).
    """
    by_bin = defaultdict(list)
    for tr in trials:
        if outcome is not None and tr.get("train_outcome") != outcome:
            continue
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        h = tr["h"]
        s, e = t0 - window_before, t0 + window_after + 1
        if s < 0 or e > len(h):
            continue
        b = tr.get("ctoa_bin")
        if b is None:
            continue
        by_bin[b].append(h[s:e])  # each: [T, N]

    labels, X_list, counts = [], [], []
    for b in sorted(by_bin.keys()):
        segs = np.stack(by_bin[b], axis=0)  # [n_trials, T, N]
        if segs.shape[0] < min_trials:
            continue
        labels.append(b)
        counts.append(segs.shape[0])
        X_list.append(segs.mean(axis=0))    # [T, N]

    if not X_list:
        return None, [], None, []

    X_cond   = np.stack(X_list, axis=0)                           # [C, T, N]
    rel_time = np.arange(-window_before, window_after + 1)        # [T]
    return X_cond, labels, rel_time, counts


def fit_jpca(X_cond, n_jpcs=1, pca_dims=6, subtract_cond_mean=True):
    """
    Fit jPCA on condition-averaged trajectories (Churchland et al., 2012).

    Steps: remove condition-independent mean -> grand-mean-subtract ->
    PCA pre-reduce -> fit dX ≈ X @ M.T -> extract skew-symmetric part M_skew
    -> eigendecompose -> orthonormalize jPC pairs.

    `subtract_cond_mean` (default True, canonical jPCA): subtract the
    cross-condition mean AT EACH TIMEPOINT, removing the condition-independent
    signal before fitting. This is essential here — the shared temporal ramp is
    a large, non-rotating drift; if left in, it dominates the dX≈MX fit and
    drives var_ratio_r2 → 0 (the rotation lives in the condition-DEPENDENT part).
    Pass False to recover the legacy (grand-mean-only) behaviour.

    Returns dict: Z [C,T,2*n_jpcs], jPCs, M_skew, M_opt, eigenvalues,
    var_ratio, var_ratio_r2, rot_index, pca_pre, X_red, grand_mean.
    """
    C, T, N = X_cond.shape
    pca_dims = min(pca_dims, N, C * T - 1)

    # ── 1. Mean-subtract ──────────────────────────────────────────────
    if subtract_cond_mean:
        # remove condition-independent (cross-condition, per-time) signal
        X_cond = X_cond - X_cond.mean(axis=0, keepdims=True)
    X_flat   = X_cond.reshape(C * T, N)
    grand_mean = X_flat.mean(axis=0)
    X_centered = X_flat - grand_mean

    # ── 2. Pre-reduce with PCA ────────────────────────────────────────
    pca_pre      = PCA(n_components=pca_dims)
    X_red_flat   = pca_pre.fit_transform(X_centered)   # [C*T, D]
    X_red        = X_red_flat.reshape(C, T, pca_dims)

    # ── 3. Fit linear dynamics via central differences ────────────────
    # Use interior time points only (avoid edge artefacts)
    X_mid  = X_red[:, 1:-1, :].reshape(-1, pca_dims)           # [C*(T-2), D]
    dX_mid = (X_red[:, 2:, :] - X_red[:, :-2, :]).reshape(-1, pca_dims) / 2

    # Least squares: dX ≈ X M_opt.T  →  M_opt.T = pinv(X) dX
    M_opt_T, _, _, _ = np.linalg.lstsq(X_mid, dX_mid, rcond=None)
    M_opt = M_opt_T.T

    # ── 4. Skew-symmetric projection ──────────────────────────────────
    M_skew = (M_opt - M_opt.T) / 2

    # ── 5. Variance metrics ───────────────────────────────────────────
    # NOTE: M_opt = M_skew + M_sym.  The cross-term <X·M_skew.T, X·M_sym.T>
    # is NOT guaranteed to be ≥ 0, so NEITHER ratio below is bounded in [0,1]:
    #
    # var_ratio (raw-dX):       = ||X·M_skew.T||² / ||dX||²
    #   Can exceed 1.0 when skew and symmetric parts of M_opt are anti-correlated
    #   in output space.  Retailed for backward-compat with permutation tests.
    #
    # var_ratio_churchland:     = ||X·M_skew.T||² / ||X·M_opt.T||²
    #   Also can exceed 1.0 for the same reason (see above).
    #   Comment "Always in [0, 1]" in earlier versions was INCORRECT.
    #
    # var_ratio_r2  ← USE THIS as the primary metric (always in [0, 1]):
    #   R² of the skew-symmetric model vs actual velocities:
    #   R² = 1 - SS_res / SS_tot  where SS_res = ||dX - X·M_skew.T||²
    #                                    SS_tot = ||dX - mean(dX)||²
    #   Answers: "what fraction of velocity variance does the rotational model explain?"
    #
    # r2_fit: R² of the full M_opt fit (sanity check, ≤ 1 guaranteed by lstsq).
    dX_hat_skew = X_mid @ M_skew.T
    dX_hat_opt  = X_mid @ M_opt.T
    var_skew    = float(np.sum(dX_hat_skew ** 2))
    var_opt     = float(np.sum(dX_hat_opt ** 2))
    var_dx_tot  = float(np.sum(dX_mid ** 2))

    var_ratio   = var_skew / var_dx_tot if var_dx_tot > 0 else 0.0
    var_ratio_churchland = var_skew / var_opt if var_opt > 0 else 0.0

    dX_mean     = dX_mid.mean(axis=0)
    ss_tot      = float(np.sum((dX_mid - dX_mean) ** 2))
    ss_res_skew = float(np.sum((dX_mid - dX_hat_skew) ** 2))
    var_ratio_r2 = max(0.0, 1.0 - ss_res_skew / ss_tot) if ss_tot > 0 else 0.0

    ss_res_opt  = float(np.sum((dX_mid - dX_hat_opt) ** 2))
    r2_fit      = max(0.0, 1.0 - ss_res_opt / ss_tot) if ss_tot > 0 else 0.0

    norm_opt   = float(np.linalg.norm(M_opt,  'fro'))
    norm_skew  = float(np.linalg.norm(M_skew, 'fro'))
    rot_index  = norm_skew / norm_opt if norm_opt > 0 else 0.0

    # ── 6. Eigendecompose M_skew → jPCs ───────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eig(M_skew)

    # Eigenvalues of a real skew-symmetric matrix are purely imaginary: ±iω.
    # Select one eigenvector per conjugate pair by keeping only those with
    # Im(λ) > 0.  This is robust to degenerate (equal) frequencies where
    # the old "take every other index after sorting" strategy could mis-pair
    # vectors across different rotation planes.
    pos_mask    = eigenvalues.imag > 0
    pos_indices = np.where(pos_mask)[0]
    pos_order   = np.argsort(-eigenvalues[pos_indices].imag)   # strongest first
    pos_indices = pos_indices[pos_order]

    if len(pos_indices) < n_jpcs:
        raise ValueError(
            f"n_jpcs={n_jpcs} requested but only {len(pos_indices)} positive-imaginary "
            "eigenvalues found in M_skew.  Reduce n_jpcs or increase pca_dims."
        )

    # Re-order the full eigenvalue array to match (for returning to caller)
    neg_mask    = eigenvalues.imag < 0
    neg_indices = np.where(neg_mask)[0]
    neg_order   = np.argsort(-np.abs(eigenvalues[neg_indices].imag))
    neg_indices = neg_indices[neg_order]
    zero_indices = np.where(~pos_mask & ~neg_mask)[0]
    full_order  = np.concatenate([pos_indices, neg_indices, zero_indices])
    eigenvalues  = eigenvalues[full_order]
    eigenvectors = eigenvectors[:, full_order]

    jPCs = np.zeros((pca_dims, 2 * n_jpcs))
    for k in range(n_jpcs):
        v = eigenvectors[:, k]               # eigenvector for +iω_k
        RI = np.column_stack([v.real, v.imag])
        # SVD gives the best orthonormal basis spanning Re(v) and Im(v)
        U, _, _ = np.linalg.svd(RI, full_matrices=False)
        jPCs[:, 2 * k]     = U[:, 0]
        jPCs[:, 2 * k + 1] = U[:, 1]

    # ── 7. Project all time points ────────────────────────────────────
    Z = (X_red_flat @ jPCs).reshape(C, T, 2 * n_jpcs)   # [C, T, 2*n_jpcs]

    # ── 8. Sign convention ────────────────────────────────────────────
    # jPC1: flip so that the mean trajectory moves in the +jPC1 direction initially
    for k in range(n_jpcs):
        z1 = Z[:, :, 2 * k]
        if np.mean(z1[:, 1] - z1[:, 0]) < 0:
            jPCs[:, 2 * k] *= -1
            Z[:, :, 2 * k] *= -1

        # jPC2: flip so that trajectories rotate counter-clockwise (CCW convention)
        # CCW: d(jPC2)/dt > 0 when jPC1 > 0  →  mean(jPC1 * d(jPC2)/dt) > 0
        # Use central differences for dz2 so shapes match z1_mid (both T-2 interior points)
        z1_mid = Z[:, 1:-1, 2 * k].flatten()
        dz2    = ((Z[:, 2:, 2 * k + 1] - Z[:, :-2, 2 * k + 1]) / 2).flatten()
        if np.mean(z1_mid * dz2) < 0:
            jPCs[:, 2 * k + 1] *= -1
            Z[:, :, 2 * k + 1] *= -1

    return {
        "Z":           Z,
        "jPCs":        jPCs,
        "M_skew":      M_skew,
        "M_opt":       M_opt,
        "eigenvalues": eigenvalues,
        "var_ratio":        var_ratio,
        "var_ratio_r2":     var_ratio_r2,
        "var_ratio_churchland": var_ratio_churchland,
        "r2_fit":       r2_fit,
        "rot_index":   rot_index,
        "pca_pre":     pca_pre,
        "X_red":       X_red,
        "grand_mean":  grand_mean,
    }


def jpca_permutation_test(X_cond, n_perms=100, pca_dims=6, n_jpcs=1, seed=0):
    """
    Permutation test for rotational structure.
    Null: shuffle time within each condition (destroys temporal order, keeps firing-rate distribution).
    Returns dict: real_var_ratio, real_rot_index, perm_*, p_var_ratio, p_rot_index, jpca_result.
    """
    rng = np.random.default_rng(seed)

    res_real = fit_jpca(X_cond, n_jpcs=n_jpcs, pca_dims=pca_dims)

    C, T, N = X_cond.shape
    perm_var_ratios  = np.zeros(n_perms)
    perm_rot_indices = np.zeros(n_perms)

    for i in range(n_perms):
        X_perm = X_cond.copy()
        for c in range(C):
            perm_idx = rng.permutation(T)
            X_perm[c] = X_perm[c, perm_idx, :]

        res_perm = fit_jpca(X_perm, n_jpcs=n_jpcs, pca_dims=pca_dims)
        perm_var_ratios[i]  = res_perm["var_ratio"]
        perm_rot_indices[i] = res_perm["rot_index"]

    real_vr = res_real["var_ratio"]
    real_ri = res_real["rot_index"]

    return {
        "real_var_ratio":   real_vr,
        "real_rot_index":   real_ri,
        "perm_var_ratios":  perm_var_ratios,
        "perm_rot_indices": perm_rot_indices,
        "p_var_ratio":      float(np.mean(perm_var_ratios  >= real_vr)),
        "p_rot_index":      float(np.mean(perm_rot_indices >= real_ri)),
        "jpca_result":      res_real,
    }


def jpca_permutation_test_condition_shuffle(X_cond, n_perms=100, pca_dims=6, n_jpcs=1, seed=0):
    """
    Condition-shuffle permutation test: shuffle conditions per neuron (preserves
    temporal structure, destroys condition-identity association).
    Returns same dict structure as jpca_permutation_test.
    """
    rng = np.random.default_rng(seed)

    res_real = fit_jpca(X_cond, n_jpcs=n_jpcs, pca_dims=pca_dims)

    C, T, N = X_cond.shape
    perm_var_ratios  = np.zeros(n_perms)
    perm_rot_indices = np.zeros(n_perms)

    for i in range(n_perms):
        X_perm = X_cond.copy()
        for n in range(N):
            perm_idx = rng.permutation(C)
            X_perm[:, :, n] = X_perm[perm_idx, :, n]

        res_perm = fit_jpca(X_perm, n_jpcs=n_jpcs, pca_dims=pca_dims)
        perm_var_ratios[i]  = res_perm["var_ratio"]
        perm_rot_indices[i] = res_perm["rot_index"]

    real_vr = res_real["var_ratio"]
    real_ri = res_real["rot_index"]

    return {
        "real_var_ratio":   real_vr,
        "real_rot_index":   real_ri,
        "perm_var_ratios":  perm_var_ratios,
        "perm_rot_indices": perm_rot_indices,
        "p_var_ratio":      float(np.mean(perm_var_ratios  >= real_vr)),
        "p_rot_index":      float(np.mean(perm_rot_indices >= real_ri)),
        "jpca_result":      res_real,
    }


# ------------------------------------------------------------------ Effective dimensionality (PR)

def participation_ratio(X, center=True):
    """Effective dimensionality (participation ratio) of a data matrix.

        PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

    where lambda_i are eigenvalues of the sample covariance. Bounded in
    [1, n_features]: PR ~ 1 means activity lies on a 1-D manifold, PR ~ n
    means an isotropic cloud.

    X: [n_samples, n_features] (e.g. T timesteps x N neurons).
    center: subtract mean across samples (standard PCA convention).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    denom = (lam ** 2).sum()
    if denom <= 0:
        return 1.0
    return float((lam.sum() ** 2) / denom)


def pr_singletrial_by_ctoa_bin(trials, align_key="target_onset",
                                window_before=15, window_after=0,
                                outcome="correct", n_components=3,
                                min_trials=5):
    """PR per CTOA bin computed on single-trial concatenated activity.

    Unlike trial-averaged PR, this fits a shared PCA basis on the
    concatenation of every single-trial segment across bins, then for
    each bin reports PR of that bin's single-trial projections
    concatenated. Trial-to-trial variability contributes spectral mass
    off PC1, so PR magnitude reflects population-level dimensionality
    closer to single-trial population analyses in the paper.
    """
    by_bin = defaultdict(list)
    for tr in trials:
        if outcome is not None and tr.get("train_outcome") != outcome:
            continue
        t0 = tr.get(align_key)
        if t0 is None:
            continue
        h = tr["h"]
        s, e = t0 - window_before, t0 + window_after + 1
        if s < 0 or e > len(h):
            continue
        b = tr.get("ctoa_bin")
        if b is None:
            continue
        by_bin[b].append(h[s:e])  # [T, N]

    labels, counts, bin_segs = [], [], {}
    for b in sorted(by_bin.keys()):
        segs = np.stack(by_bin[b], axis=0)
        if segs.shape[0] < min_trials:
            continue
        labels.append(b)
        counts.append(segs.shape[0])
        bin_segs[b] = segs

    if not labels:
        return None

    flat = np.concatenate(
        [bin_segs[b].reshape(-1, bin_segs[b].shape[2]) for b in labels], axis=0
    )
    N = flat.shape[1]
    k = min(n_components, N, flat.shape[0])
    pca = PCA(n_components=k).fit(flat)

    PR = []
    for b in labels:
        flat_b = bin_segs[b].reshape(-1, N)
        Z = pca.transform(flat_b)
        PR.append(participation_ratio(Z))

    bin_to_ms = defaultdict(list)
    for tr in trials:
        if tr.get("train_outcome") != outcome:
            continue
        b, ms = tr.get("ctoa_bin"), tr.get("ctoa_ms")
        if b is not None and ms is not None and b in labels:
            bin_to_ms[b].append(ms)
    ctoa_ms_mean = np.array([np.mean(bin_to_ms[b]) if b in bin_to_ms else np.nan
                              for b in labels])

    return {
        "PR":           np.array(PR),
        "labels":       labels,
        "counts":       counts,
        "n_components": k,
        "explained":    float(pca.explained_variance_ratio_.sum()),
        "ctoa_ms_mean": ctoa_ms_mean,
    }


# ------------------------------------------------------------------ Trajectory tangling

def compute_tangling(X_cond, epsilon=1e-3):
    """
    Trajectory tangling Q(t) per Russo et al. (2018).
    Q(t,c) = max_{t',c'} ||dx-dx'||² / (||x-x'||² + epsilon)
    High Q means similar positions but different velocities (crossed trajectories).
    Returns Q [C, T].
    """
    C, T, N = X_cond.shape

    dX = np.zeros_like(X_cond)
    dX[:, 1:-1, :] = (X_cond[:, 2:, :] - X_cond[:, :-2, :]) / 2
    dX[:, 0, :]    = X_cond[:, 1, :]  - X_cond[:, 0, :]
    dX[:, -1, :]   = X_cond[:, -1, :] - X_cond[:, -2, :]

    X_flat  = X_cond.reshape(C * T, N)
    dX_flat = dX.reshape(C * T, N)

    # pairwise ||x_i - x_j||^2 via expansion: ||a||^2 + ||b||^2 - 2 a·b
    X_sq  = np.sum(X_flat  ** 2, axis=1)
    dX_sq = np.sum(dX_flat ** 2, axis=1)
    dist_x  = np.maximum(X_sq[:, None]  + X_sq[None, :]  - 2 * (X_flat  @ X_flat.T),  0.0)
    dist_dx = np.maximum(dX_sq[:, None] + dX_sq[None, :] - 2 * (dX_flat @ dX_flat.T), 0.0)

    ratio = dist_dx / (dist_x + epsilon)
    np.fill_diagonal(ratio, 0.0)
    Q_flat = ratio.max(axis=1)

    return Q_flat.reshape(C, T)


def tangling_by_ctoa_bin(trials, align_key="target_onset",
                          window_before=15, window_after=30,
                          pca_dims=None, outcome="correct",
                          epsilon=1e-3, min_trials=5):
    """
    Tangling per CTOA bin. pca_dims pre-reduces before tangling (avoids
    curse of dimensionality; try pca_dims=20).
    Returns dict: Q, Q_mean, labels, rel_time, counts, X_cond, ctoa_ms_mean.
    """
    X_cond, labels, rel_time, counts = prepare_jpca_input(
        trials, align_key=align_key,
        window_before=window_before, window_after=window_after,
        min_trials=min_trials, outcome=outcome,
    )
    if X_cond is None:
        return None

    C, T, N = X_cond.shape

    if pca_dims is not None and pca_dims < N:
        X_flat = X_cond.reshape(C * T, N)
        pca = PCA(n_components=pca_dims)
        X_cond = pca.fit_transform(X_flat).reshape(C, T, pca_dims)

    Q = compute_tangling(X_cond, epsilon=epsilon)

    bin_to_ms = defaultdict(list)
    for tr in trials:
        if tr.get("train_outcome") != outcome:
            continue
        b = tr.get("ctoa_bin")
        ms = tr.get("ctoa_ms")
        if b is not None and ms is not None and b in labels:
            bin_to_ms[b].append(ms)
    ctoa_ms_mean = np.array([np.mean(bin_to_ms[b]) if b in bin_to_ms else np.nan
                              for b in labels])

    return {
        "Q":            Q,
        "Q_mean":       Q.mean(axis=1),
        "labels":       labels,
        "rel_time":     rel_time,
        "counts":       counts,
        "X_cond":       X_cond,
        "ctoa_ms_mean": ctoa_ms_mean,
    }


def polynomial_regression(x, y, degree=1):
    """Polyfit + F-test vs intercept-only null. Returns dict: r2, p_value, coeffs, y_hat."""
    from scipy import stats as sp_stats

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n <= degree + 1:
        return {"r2": np.nan, "p_value": np.nan, "coeffs": None,
                "y_hat": None, "degree": degree}

    coeffs = np.polyfit(x, y, degree)
    y_hat  = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    k = degree  # F-test: k predictors vs intercept-only
    df1 = k
    df2 = n - k - 1
    if df2 <= 0 or ss_tot == 0:
        p_value = np.nan
    else:
        F = ((ss_tot - ss_res) / df1) / (ss_res / df2)
        p_value = float(1 - sp_stats.f.cdf(F, df1, df2))

    return {"r2": r2, "p_value": p_value, "coeffs": coeffs,
            "y_hat": y_hat, "degree": degree, "x": x, "y": y}


def plot_spatial_map(model, h_mean, title="Spatial map"):
    """Plot mean activity [n_tot] on the 2D sheet.

    Works for both the base model (separate `e_coords`/`i_coords` buffers in
    integer grid units) and the topo model (single `coords` buffer in [-1,1]);
    the latter is mapped back to integer grid units so the shared plotting code
    below is unchanged.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if hasattr(model, "e_coords"):
        e_coords = model.e_coords.cpu().numpy()
        i_coords = model.i_coords.cpu().numpy()
    else:
        coords = model.coords.cpu().numpy()
        n_exc = model.n_exc
        side = int(round(np.sqrt(n_exc)))
        denom = max(side - 1, 1)
        e_coords = (coords[:n_exc] + 1.0) / 2.0 * denom   # [-1,1] -> [0, side-1]
        i_coords = (coords[n_exc:] + 1.0) / 2.0 * denom
    n_exc = e_coords.shape[0]

    h_np = h_mean.detach().cpu().numpy()
    h_e = h_np[:n_exc]
    h_i = h_np[n_exc:]

    side = int(np.sqrt(n_exc))
    grid_e = np.zeros((side, side))
    for k, (x, y) in enumerate(e_coords):
        grid_e[int(round(y)), int(round(x))] = h_e[k]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(grid_e, cmap="RdBu_r", origin="lower", aspect="equal")
    axes[0].set_title(f"{title} — E neurons ({side}\u00d7{side} grid)")
    plt.colorbar(im, ax=axes[0])

    sc = axes[1].scatter(i_coords[:, 0], i_coords[:, 1],
                         c=h_i, cmap="RdBu_r", s=60, edgecolors="k", lw=0.5)
    axes[1].set_xlim(-0.5, side - 0.5)
    axes[1].set_ylim(-0.5, side - 0.5)
    axes[1].set_title(f"{title} — I neurons (random positions)")
    plt.colorbar(sc, ax=axes[1])

    plt.tight_layout()
    return fig


def plot_connectivity_matrix(model):
    """Visualises rec_mask: full matrix and per-block (E/E, I/E, E/I, I/I) density.
    Also prints scaled tau diagnostics (Chen & Gong 2022 scaling).
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    mask = model.rec_mask.cpu().numpy()
    n_exc = model.n_exc

    # --- tau scaling diagnostics ---
    side = int(np.sqrt(n_exc))
    max_dist = math.sqrt(2) * (side - 1)
    ref_max_dist = 88.0
    scale = max_dist / ref_max_dist

    tau_ee = getattr(model, '_tau_ee', 8.0)
    tau_ie = getattr(model, '_tau_ie', 10.0)
    tau_ei = getattr(model, '_tau_ei', 20.0)
    tau_ii = getattr(model, '_tau_ii', 20.0)

    tau_ee_s = tau_ee * scale
    tau_ie_s = tau_ie * scale
    tau_ei_s = tau_ei * scale
    tau_ii_s = tau_ii * scale

    print(f"Grid: {side}x{side}  max_dist = {max_dist:.1f} nodes  scale = {scale:.3f}")
    print(f"tau_EE: {tau_ee:.1f} → {tau_ee_s:.2f}  |  p(max_dist) = {math.exp(-max_dist/tau_ee_s):.4f}  (target < 0.05)")
    print(f"tau_IE: {tau_ie:.1f} → {tau_ie_s:.2f}  |  p(max_dist) = {math.exp(-max_dist/tau_ie_s):.4f}")
    print(f"tau_EI: {tau_ei:.1f} → {tau_ei_s:.2f}  |  p(max_dist) = {math.exp(-max_dist/tau_ei_s):.4f}")
    print(f"tau_II: {tau_ii:.1f} → {tau_ii_s:.2f}  |  p(max_dist) = {math.exp(-max_dist/tau_ii_s):.4f}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(mask, cmap="Greys", aspect="auto")
    axes[0].axhline(n_exc - 0.5, color="red", lw=0.8)
    axes[0].axvline(n_exc - 0.5, color="red", lw=0.8)
    axes[0].set_title("Full rec_mask")
    axes[0].set_xlabel("Sender neuron")
    axes[0].set_ylabel("Receiver neuron")

    blocks = {
        "E\u2192E": mask[:n_exc, :n_exc],
        "I\u2192E": mask[:n_exc, n_exc:],
        "E\u2192I": mask[n_exc:, :n_exc],
        "I\u2192I": mask[n_exc:, n_exc:],
    }
    labels = list(blocks.keys())
    densities = [v.mean() for v in blocks.values()]
    axes[1].bar(labels, densities, color=["steelblue", "coral", "steelblue", "coral"])
    axes[1].set_ylabel("Connection density")
    axes[1].set_title("Density by block")

    plt.tight_layout()
    return fig
